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
from nav2_msgs.srv import ClearEntireCostmap
from simulation_launch.action import SendGoalToNav2

from typing import Optional
import numpy as np

class PlannerConfigManager(Node):
    def __init__(self):
        super().__init__('planner_config_manager')
        self.cb_group = ReentrantCallbackGroup()
        # TODO - using ROS2 params for now as a quick and dirty hack for communication, but this should 
        # really be ported to a service or action server:
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('goal_yaw', 0.0)
        self.declare_parameter('xy_tolerance', 0.25)
        self.declare_parameter('yaw_tolerance', 0.40)
        self.declare_parameter('max_episode_steps', 500)
        self.declare_parameter('collision_threshold', 0.20)
        self.declare_parameter('planner_type', 'dwb')
        self.declare_parameter('lidar_downsample_n', 10)
        # Initialize objects for transform processing:
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Configure subscribers:
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.process_scan, 10, callback_group=self.cb_group)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.process_odom, 10, callback_group=self.cb_group)
        # Configure publishers:
        self.obs_pub      = self.create_publisher(Float32MultiArray, '/rl/observation', 10)
        self.status_pub   = self.create_publisher(String, '/rl/status', 10)
        self.set_pose_pub = self.create_publisher(Pose2D, '/simulation/set_pose', 10)  # NOTE - this is what diff_drive_model subscribes to for snapping internal pose to episode start pose
        # Provide service for RL pipeline to reset planner configuration at start of new episode:
        self.reset_srv = self.create_service(Empty, '/rl/reset', self.reset_config, callback_group=self.cb_group)
        # Configure service clients:
        self.ctrl_param_client = self.create_client(SetParameters, '/controller_server/set_parameters', callback_group=self.cb_group)
        self.local_costmap_client = self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap', callback_group=self.cb_group)
        self.global_costmap_client = self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap', callback_group=self.cb_group)
        # Configure action client for the Nav2 goal manager:
        self.goal_action_client = ActionClient(self, SendGoalToNav2, 'send_goal_to_nav2', callback_group=self.cb_group)
        # Configure timer for periodically publishing RL pipeline state vector (observation):
        self.obs_timer = self.create_timer(0.05, self.publish_observation, callback_group=cb_group)
        # Internal episode state:
        self.episode_status: str = "idle"
        self.episode_step: int = 0
        self.curr_goal_x: float = 0.0
        self.curr_goal_y: float = 0.0
        self.curr_goal_yaw: float = 0.0
        self.latest_ranges: Optional[np.ndarray] = None
        self.latest_vx: float = 0.0
        self.latest_vy: float = 0.0
        self.latest_wz: float = 0.0

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
        for i in range(1, n_samples+1):
            j = 2*(i-1)
            scan_segment = raw_scan[(i-1):i]
            min_idx = np.where(scan_segment == np.min(scan_segment))[0][0]  # will grab the lowest indexed feature if there are multiple min returns
            feature_range = scan_segment[min_idx]
            min_segment_brng = (i-1)*(2*np.pi/n_samples)
            max_segment_brng = (i)*(2*np.pi/n_samples)
            feature_bearing = min_segment_brng + min_idx*((max_segment_brng-min_segment_brng)/scan_segment.shape[0])  # extract feature bearing from segment bounds using feature index
            downsampled_scan[j] = feature_range
            downsampled_scan[j+1] = feature_bearing
        self.latest_ranges = downsampled_scan

    def process_odom(self, msg: Odometry) -> None:
        # TODO - confirm that I'm interpretting the Twist properly:
        self.latest_vx = msg.twist.twist.linear.x   # velocity in x direction
        self.latest_vy = msg.twist.twist.linear.y   # velocity in y direction
        self.latest_wz = msg.twist.twist.angular.z  # angular velocity about yaw axis

    def reset_config(self):
        # TODO - fill this out
        pass

    def publish_observation(self):
        # TODO - fill this out
        pass