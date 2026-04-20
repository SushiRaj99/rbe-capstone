#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Pose2D
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters
from nav2_msgs.srv import ClearEntireCostmap, LoadMap
from simulation_launch.action import SendGoalToNav2
import rl_pipeline.pipeline_utils as putils

from typing import Optional, Tuple
import numpy as np
import os, time

class PlannerConfigManager(Node):
    def __init__(self):
        super().__init__('planner_config_manager')
        self.cb_group = ReentrantCallbackGroup()
        # TODO - using ROS2 params for now as a quick and dirty hack for communication, but this should 
        # really be ported to a service or action server:
        self.declare_parameter('goal_id', '')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('xy_tolerance', 0.25)
        self.declare_parameter('yaw_tolerance', 0.40)
        self.declare_parameter('max_episode_steps', 500)
        self.declare_parameter('collision_threshold', putils.LIDAR_COLLISION_RANGE) # defines min lidar distance that triggers a 'collision' (TODO - might need to disable collision monitor for this)
        self.declare_parameter('planner_type', 'dwb')                               # placeholder for adding MPPI later
        self.declare_parameter('lidar_downsample_n', putils.N_LIDAR_RAYS)
        self.declare_parameter('map_filepath', '')
        # Initialize objects for transform processing:
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Configure subscribers:
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.process_scan, 10, callback_group=self.cb_group)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.process_odom, 10, callback_group=self.cb_group)
        self.action_sub = self.create_subscription(Float32MultiArray, '/rl/action', self.process_action, 10, callback_group=self.cb_group)
        # Configure publishers:
        self.obs_pub = self.create_publisher(Float32MultiArray, '/rl/observation', 10)
        self.status_pub = self.create_publisher(String, '/rl/status', 10)
        self.set_pose_pub = self.create_publisher(Pose2D, '/simulation/set_pose', 10)  # NOTE - this is what diff_drive_model subscribes to for snapping internal pose to episode start pose
        # Provide service for RL pipeline to reset planner configuration at start of new episode:
        self.reset_srv = self.create_service(Empty, '/rl/reset', self.reset_config, callback_group=self.cb_group)
        # Configure service clients:
        self.load_map_client = self.create_client(LoadMap, '/map_server/load_map', callback_group=self.cb_group)
        self.local_costmap_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap', callback_group=self.cb_group)
        self.global_costmap_client = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap', callback_group=self.cb_group)
        self.ctrl_param_client = self.create_client(SetParameters, '/controller_server/set_parameters', callback_group=self.cb_group)
        # Configure action client for the Nav2 goal manager:
        self.goal_action_client = ActionClient(self, SendGoalToNav2, 'send_goal_to_nav2', callback_group=self.cb_group)
        # Configure timer for periodically publishing RL pipeline state vector (observation):
        self.obs_timer = self.create_timer(0.05, self.publish_observation, callback_group=self.cb_group)
        # Internal episode state:
        self.episode_status: str = "idle"
        self.episode_name: str = ""
        self.episode_step: int = 0
        self.curr_goal_x: float = 0.0
        self.curr_goal_y: float = 0.0
        self.curr_goal_yaw: float = 0.0
        self.curr_goal_reached: bool = False
        self.latest_ranges: Optional[np.ndarray] = None
        self.latest_vx: float = 0.0     # only need to consider forward and steering velocities for differential drive robots
        self.latest_wz: float = 0.0     # only need to consider forward and steering velocities for differential drive robots

    def process_scan(self, msg: LaserScan) -> None:
        # In an attempt to prevent important features from being "sampled out", the downsampled scan vector 
        # will actually include N (range, bearing) pairs flattened into a 2*N array (where N is the number of
        # scan samples to include in the vector). So instead of including the given scans at fixed intervals
        # (of total_scans/N), all scans will be sliced into N segments, and the scan with the closest range 
        # return from within a given sector, along with it's respective bearing, will saved. This ensures that 
        # the "most important" (closest) features make it into the state vector used by the RL pipeline: 
        n_samples = self.get_parameter('lidar_downsample_n').value
        raw_scan = np.array(msg.ranges, dtype=np.float32)
        raw_scan = np.where(np.isfinite(raw_scan), raw_scan, msg.range_max) # fills in non-finite values of raw_scan with the msg.range_max
        downsampled_scan = np.zeros((int(2*n_samples),), dtype=np.float32)
        lasers_per_segment = int(len(raw_scan) / n_samples)
        for i in range(1, n_samples+1):
            j = 2*(i-1)
            scan_segment = raw_scan[((i-1)*lasers_per_segment):(i*lasers_per_segment)]
            min_idx = np.where(scan_segment == np.min(scan_segment))[0][0]  # will grab the lowest indexed feature if there are multiple min returns
            feature_range = scan_segment[min_idx]
            min_segment_brng = (i-1)*(2*np.pi/n_samples)
            max_segment_brng = (i)*(2*np.pi/n_samples)
            feature_bearing = min_segment_brng + min_idx*((max_segment_brng-min_segment_brng)/scan_segment.shape[0])  # extract feature bearing from segment bounds using feature index
            downsampled_scan[j] = feature_range
            downsampled_scan[j+1] = feature_bearing
        self.latest_ranges = downsampled_scan

    def process_odom(self, msg: Odometry) -> None:
        self.latest_vx = msg.twist.twist.linear.x   # forward velocity
        self.latest_wz = msg.twist.twist.angular.z  # steering velocity

    def process_action(self, msg: Float32MultiArray) -> None:
        # This method serves as the "middle-man" between the RL pipeline and the Nav2 stack for 
        # denormalizing and setting planner parameters. Especially considering that the SB3 actions (PPO 
        # output layer) are pushed through a tanh() layer to normalize outputs in range [-1, 1]. So the 
        # action input to this method must be mapped from [-1, 1] -> [0, 1] -> [lo, hi] per the associated 
        # parameter descriptor in the pipeline_utils module:
        if self.episode_status == 'idle':
            return
        planner_type = self.get_parameter('planner_type').value
        param_spec = putils.PLANNER_PARAM_BOUNDS.get(planner_type, putils.DWB_PARAM_BOUNDS) # defaults to DWB param list
        normalized_params = np.asarray(msg.data, dtype=np.float32)
        if len(normalized_params) != len(param_spec):
            self.get_logger().error(
                f"Action dimension mismatch: received {len(normalized_params)}, " +
                f"expected {len(param_spec)} for planner '{planner_type}'"
            )
            return
        if not self.ctrl_param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("controller_server/set_parameters not available — planner params not updated this step.")
            return
        normalized_values = (normalized_params + 1.0) / 2.0  # converts [-1, 1] -> [0, 1]
        nav2_params = []
        for (param_name, lo, hi), norm_val in zip(param_spec, normalized_values):
            real_val = float(np.clip(norm_val, 0.0, 1.0) * (hi - lo) + lo)
            nav2_params.append(putils.make_rclparam_double(param_name, real_val))
            self.get_logger().debug(f"{param_name}: converted {norm_val} to {real_val}")
        request = SetParameters.Request()
        request.parameters = nav2_params
        future = self.ctrl_param_client.call_async(request)
        putils.spin_wait_for_future(future, timeout=2.0)

    def reset_config(self, request: Empty.Request, response: Empty.Response) -> Empty.Response:
        # This method serves as the reset service callback. All of the applicable ROS2 parameters should be set prior 
        # to calling the reset service:
        self.get_logger().info("Episode reset triggered.")
        self.episode_name = self.get_parameter('goal_id').value
        start_x = self.get_parameter('start_x').value
        start_y = self.get_parameter('start_y').value
        start_yaw = self.get_parameter('start_yaw').value
        self.curr_goal_reached = False
        self.curr_goal_x = self.get_parameter('goal_x').value
        self.curr_goal_y = self.get_parameter('goal_y').value
        self.curr_goal_yaw = self.get_parameter('goal_yaw').value
        xy_tol = self.get_parameter('xy_tolerance').value
        yaw_tol = self.get_parameter('yaw_tolerance').value
        map_filepath = self.get_parameter('map_filepath').value
        # Only hot-swap the map if a new one was requested (empty map_filepath 
        # means to keep the same map):
        if (map_filepath.strip()):
            self.load_map(map_filepath)
        # Reset the diff_drive_model to the new start pose:
        self.reset_diff_drive(start_x, start_y, start_yaw)
        # Clear stale occupancy data from both costmaps:
        self.clear_costmaps()
        # Reset internal episode bookkeeping:
        self.episode_step = 0
        self.episode_status = 'running'
        self.latest_ranges = None   # discard lingering scans from previous episode
        # Issue new Nav2 goal via goal_manager (non-blocking):
        self.send_nav2_goal(self.curr_goal_x, self.curr_goal_y, self.curr_goal_yaw, xy_tol, yaw_tol)
        self.get_logger().info(
            f"Episode started: map={map_filepath or '(unchanged)'} " +
            f"start=({start_x:.2f}, {start_y:.2f}, {start_yaw:.2f}) " +
            f"goal=({self.curr_goal_x:.2f}, {self.curr_goal_y:.2f}, {self.curr_goal_yaw:.2f})"
        )
        return response

    def load_map(self, map_filepath: str) -> None:
        if not os.path.isfile(map_filepath):
            self.get_logger().warn(f"Map file not found: {map_filepath} - keeping previous map.")
            return
        if not self.load_map_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/map_server/load_map service not available - keeping previous map.")
            return
        request = LoadMap.Request()
        request.map_url = map_filepath
        future = self.load_map_client.call_async(request)
        putils.spin_wait_for_future(future, timeout=5.0)
        if not future.done() or future.result() is None:
            self.get_logger().warn(f"load_map() timed out for '{map_filepath}' - keeping previous map.")
            return
        if future.result().result != 0: 
            self.get_logger().warn(
                f"load_map() returned error code {future.result().result} " +
                f"for '{map_filepath}' - keeping previous map."
            )
        else:
            self.get_logger().info(f"Map loaded: {map_filepath}")

    def reset_diff_drive(self, x: float, y: float, yaw: float) -> None:
        msg = Pose2D()
        msg.x = x
        msg.y = y
        msg.theta = yaw
        self.set_pose_pub.publish(msg)
        time.sleep(0.05)    # let diff_drive_model process the pose before Nav2 starts
        self.get_logger().debug(f"diff_drive_model pose reset to {x:.3f}, {y:.3f}, {yaw:.3f}")

    def clear_costmaps(self) -> None:
        for client, name in [(self.local_costmap_client, 'local'), (self.global_costmap_client, 'global')]:
            if client.wait_for_service(timeout_sec=1.0):
                client.call_async(ClearEntireCostmap.Request())
            else:
                self.get_logger().warn(f"{name} costmap clear service not available.")

    def send_nav2_goal(self, x: float, y: float, yaw: float, xy_tol: float, yaw_tol: float) -> None:
        if not self.goal_action_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn("send_goal_to_nav2 action server not ready.")
            return
        goal_msg = SendGoalToNav2.Goal()
        goal_msg.goal_id = self.episode_name
        goal_msg.x = x
        goal_msg.y = y
        goal_msg.yaw = yaw
        goal_msg.xy_tolerance = xy_tol
        goal_msg.yaw_tolerance = yaw_tol
        try:
            #self.curr_goal_reached = False     # This should be done as soon as /rl/reset service is triggered (within reset_config())
            future = self.goal_action_client.send_goal_async(goal_msg)   # intentionally non-blocking
            future.add_done_callback(self.process_goalmanager_response)
        except Exception as e:
            self.get_logger().error(f"Failed to send goal: {e}")
    
    def process_goalmanager_response(self, future: Future) -> None:
        # Need to ensure early termination within this callback to ensure thread safety 
        # during initialization (no blocking):
        if not rclpy.ok() or self.context is None:
            return
        try:
            goal_ptr: ClientGoalHandle = future.result()
        except Exception as e:
            self.get_logger().error(f"SendGoalToNav2 response failed: {e}")
            return
        # If the goal handle was acquired check the status of the request:
        if not goal_ptr.accepted:
            self.get_logger().warn("SendGoalToNav2 request rejected")
            return
        self.get_logger().info("SendGoalToNav2 request accepted")
        # Attempt to configure the postprocessing for the SendGoalToNav2 server:
        try:
            future_result = goal_ptr.get_result_async()
            future_result.add_done_callback(self.process_goalmanager_result)
        except Exception as e:
            self.get_logger().error(f"Failed to configure SendGoalToNav2 post processing: {e}")

    def process_goalmanager_result(self, future: Future) -> None:
        # Need to ensure early termination within this callback to ensure thread safety 
        # during initialization (no blocking):
        if not rclpy.ok() or self.context is None:
            return
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"Failed to acquire result from SendGoalToNav2: {e}")
            return
        # Record success:
        self.get_logger().info(f"Goal finished: GoalManager (goal_reached={result.goal_reached}), Nav2 (status={result.nav2_status})")
        self.curr_goal_reached = True

    def publish_observation(self) -> None:
        # Don't publish anything until an episode has been started via /rl/reset service provided by this node:
        if self.episode_status == 'idle':
            return
        # Increment step counter and freeze status once terminal is detected. The status is not re-evaluated after 
        # terminal so that the RL env sees the same terminal flag on every observation until the next reset:
        if self.episode_status == 'running':
            self.episode_step += 1
            self.episode_status = self.check_terminal()
        observation = self.build_observation()
        if observation is not None:
            msg = Float32MultiArray()
            msg.data = observation.tolist()
            self.obs_pub.publish(msg)
        status_msg = String()
        status_msg.data = self.episode_status
        self.status_pub.publish(status_msg)

    def check_terminal(self) -> str:
        collision_thresh = self.get_parameter('collision_threshold').value
        max_steps = self.get_parameter('max_episode_steps').value
        xy_tol = self.get_parameter('xy_tolerance').value
        yaw_tol = self.get_parameter('yaw_tolerance').value
        # Check for collision:
        if (self.latest_ranges is not None) and (np.min(self.latest_ranges) < collision_thresh):
            return 'collision'
        # Check if the goal was reached:
        """
        # Shouldn't actually need to acquire/compute robot pose here since this should all be handled by SendGoal2Nav2: 
        pose = self.get_robot_pose()
        if pose is not None:
            x, y, yaw = pose
            xy_err = np.sqrt((x - self.curr_goal_x)**2 + (y - self.curr_goal_y)**2)
            yaw_err = abs(
                ((yaw - self.curr_goal_yaw) + np.pi) % (2.0 * np.pi) - np.pi)
            if xy_err < xy_tol and yaw_err < yaw_tol:
                return 'goal_reached'"""
        if (self.curr_goal_reached):
            return 'goal_reached'
        # Check for "timeout" (max number of steps exceeded):
        if self.episode_step >= max_steps:
            return 'timeout'
        # If no terminal conditions were identified, the episode is still running:
        return 'running'

    def build_observation(self) -> Optional[np.ndarray]:
        # Assembles the raw observation (state) vector:
        #   [
        #       lidar_range_0,
        #       lidar_bearing_0,
        #       ...
        #       lidar_range_N,
        #       lidar_bearing_N,
        #       error_from_goal_x,
        #       error_from_goal_y,
        #       error_from_goal_yaw,
        #       forward_velocity,
        #       steering_velocity
        #   ]
        observation = None
        if (self.latest_ranges is not None):
            pose = self.get_robot_pose()
            if pose is not None:
                x, y, yaw = pose
                delta_x = self.curr_goal_x - x
                delta_y = self.curr_goal_y - y
                # The bearing to goal should be expressed in the rohbot frame, wrapped to [-pi, pi]
                world_angle_to_goal = np.arctan2(delta_y, delta_x)
                delta_yaw = float( ((world_angle_to_goal - yaw) + np.pi) % (2.0*np.pi) - np.pi )
                observation = np.concatenate([
                    self.latest_ranges,
                    np.array([
                        delta_x,
                        delta_y,
                        delta_yaw,
                        self.latest_vx,
                        self.latest_wz
                    ])
                ])
        return observation

    def get_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = float(np.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            ))
            return x, y, yaw
        except Exception as e:
            self.get_logger().warn("Unable to access transform buffer")
            return None

def main(args=None):
    rclpy.init()
    node = PlannerConfigManager()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()