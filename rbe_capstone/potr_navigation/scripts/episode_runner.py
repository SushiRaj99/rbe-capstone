#!/usr/bin/env python3
import rclpy
import copy
import json
import math
import os
import time
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.action.client import ClientGoalHandle
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import LoadMap
from geometry_msgs.msg import PoseStamped, Pose2D, Pose
from tf2_ros import Buffer, TransformListener
from std_srvs.srv import Trigger
from simulation_launch.action import SendGoalToNav2
from lifecycle_msgs.srv import GetState
from potr_navigation.srv import GetMetrics, SwitchPlanner, SetParamPreset
from potr_navigation.msg import EpisodeMetrics
from ament_index_python.packages import get_package_share_directory

from typing import Tuple, Optional

RUN_CONFIGS = [
    ('DWB',  1),
    ('DWB',  2),
    ('MPPI', 1),
    ('MPPI', 2),
]

# Run loop states
S_INIT              = 'INIT'
S_SWITCHING_PLANNER = 'SWITCHING_PLANNER'
S_START_EPISODE     = 'START_EPISODE'
S_MAP_SWITCHING     = 'MAP_SWITCHING'
S_SETTLING          = 'SETTLING'
S_GOAL_ACTIVE       = 'GOAL_ACTIVE'
S_GETTING_METRICS   = 'GETTING_METRICS'
S_WAITING_FOR_RL    = 'WAITING_FOR_RL'
S_DONE              = 'DONE'

SETTLE_SECS = 2.0   # seconds to wait after respawn before sending goal


def decode_nav2_status(status_code: int) -> str:
    decoder = {
        0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING",
        3: "CANCELING", 4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED",
    }
    return decoder.get(status_code, f"UNKNOWN_CODE_{status_code}")


class EpisodeRunner(Node):
    def __init__(self):
        super().__init__('episode_runner')
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter('default_map', 'mixed')
        self.declare_parameter('map_to_odom_x', -4.0)   # map->odom static TF x translation
        self.declare_parameter('map_to_odom_y',  0.0)
        self.declare_parameter('episodes', '')
        self.declare_parameter('rl_mode', False)

        # Nav2 action client and external goal action server
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav_client.wait_for_server()

        self.action_server = ActionServer(
            self, SendGoalToNav2, 'send_goal_to_nav2',
            self.manage_send_goal, callback_group=self.cb_group
        )
        self.reset_action_goal_state()
        self.pending_result = None
        self.sendgoal_success = False
        self.manual_nav2_override = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Clients
        self.nav2_state_client     = self.create_client(GetState,       '/bt_navigator/get_state')
        self.reset_client          = self.create_client(Trigger,        '/potr_navigation/reset_metrics',   callback_group=self.cb_group)
        self.metrics_client        = self.create_client(GetMetrics,     '/potr_navigation/get_metrics',     callback_group=self.cb_group)
        self.switch_planner_client = self.create_client(SwitchPlanner,  '/potr_navigation/switch_planner',  callback_group=self.cb_group)
        self.set_preset_client     = self.create_client(SetParamPreset, '/potr_navigation/set_param_preset', callback_group=self.cb_group)
        self.load_map_client       = self.create_client(LoadMap,        '/map_server/load_map',             callback_group=self.cb_group)

        # Publishers
        self.set_pose_pub     = self.create_publisher(Pose2D,          '/simulation/set_pose', 10)
        self.metrics_pub      = self.create_publisher(EpisodeMetrics,  '/potr_navigation/episode_metrics', 10)
        self.current_goal_pub = self.create_publisher(Pose,            '/potr_navigation/current_goal', 10)  # read by metrics_tracker
        self.create_service(Trigger, '/potr_navigation/start_episode', self.handle_start_episode, callback_group=self.cb_group)

        # Run loop state
        self.nav2_active      = False
        self.rl_state         = S_INIT
        self.rl_run_index     = 0
        self.rl_episode_index = 0
        self.rl_episodes      = []
        self.rl_current_map   = None
        self.rl_settle_until  = None
        self.rl_nav_status    = None   # set by goal result callback
        self.rl_switch_called = False  # prevents re-entry while waiting for planner switch callback

        # Action server goal tracking (separate from run loop)
        self.action_nav2_goal_ptr = None
        self.action_nav2_status   = None

        self.create_timer(0.1, self.check_nav2_active, callback_group=self.cb_group)
        self.create_timer(0.5, self.run_loop_tick,     callback_group=self.cb_group)
        self.create_timer(0.1, self.monitor_goal,      callback_group=self.cb_group)

        self.get_logger().info(f'Episode runner ready — {len(RUN_CONFIGS)} run configs')

    def run_loop_tick(self) -> None:
        if not self.nav2_active:
            return

        if self.rl_state == S_INIT:
            episodes_str = self.get_parameter('episodes').value
            self.get_logger().info(f'episodes param length: {len(episodes_str)} chars')
            if not episodes_str.strip():
                self.get_logger().error('episodes param is empty — check goalpoints_episode.yaml is installed and loaded')
                self.rl_state = S_DONE
                return
            try:
                self.rl_episodes = json.loads(episodes_str)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Failed to parse episodes JSON: {e}')
                self.rl_state = S_DONE
                return
            rl_mode = self.get_parameter('rl_mode').value
            if rl_mode:
                self.get_logger().info(
                    f'RL mode: {len(self.rl_episodes)} episodes, waiting for start_episode calls'
                )
                self.rl_state = S_WAITING_FOR_RL
            else:
                self.get_logger().info(
                    f'Run loop starting: {len(RUN_CONFIGS)} configs x {len(self.rl_episodes)} episodes'
                )
                self.rl_state = S_SWITCHING_PLANNER
                self.rl_switch_called = False

        elif self.rl_state == S_WAITING_FOR_RL:
            pass

        elif self.rl_state == S_SWITCHING_PLANNER:
            if self.rl_switch_called:
                return
            if (not self.switch_planner_client.service_is_ready() or
                    not self.set_preset_client.service_is_ready()):
                self.get_logger().info('Waiting for planner controller services...')
                return
            self.rl_switch_called = True
            self.switch_to_config()

        elif self.rl_state == S_START_EPISODE:
            if self.rl_episode_index >= len(self.rl_episodes):
                self.rl_run_index += 1
                if self.rl_run_index >= len(RUN_CONFIGS):
                    self.get_logger().info('All runs complete!')
                    self.rl_state = S_DONE
                    return
                self.rl_episode_index = 0
                self.rl_state = S_SWITCHING_PLANNER
                self.rl_switch_called = False
                return
            episode = self.rl_episodes[self.rl_episode_index]
            map_name = episode.get('map', self.get_parameter('default_map').value)
            if map_name != self.rl_current_map:
                self.rl_state = S_MAP_SWITCHING
                self.do_map_switch(map_name)
            else:
                self.do_respawn(episode['start'])
                self.rl_settle_until = self.get_clock().now() + rclpy.duration.Duration(seconds=SETTLE_SECS)
                self.rl_state = S_SETTLING

        elif self.rl_state == S_SETTLING:
            if self.rl_settle_until is not None and self.get_clock().now() >= self.rl_settle_until:
                episode = self.rl_episodes[self.rl_episode_index]
                future = self.reset_client.call_async(Trigger.Request())
                future.add_done_callback(lambda f: self.on_episode_reset_done(f, episode))
                self.rl_state = S_GOAL_ACTIVE  # block re-entry until goal done

        elif self.rl_state == S_GOAL_ACTIVE:
            if self.rl_nav_status is not None:
                self.rl_state = S_GETTING_METRICS
                future = self.metrics_client.call_async(GetMetrics.Request())
                future.add_done_callback(self.on_episode_metrics_done)

    def switch_to_config(self) -> None:
        planner, preset = RUN_CONFIGS[self.rl_run_index]
        self.get_logger().info(
            f'Run {self.rl_run_index + 1}/{len(RUN_CONFIGS)}: switching to {planner} preset {preset}'
        )
        req = SwitchPlanner.Request()
        req.planner_name = planner
        future = self.switch_planner_client.call_async(req)
        future.add_done_callback(lambda f: self.on_switch_planner_done(f, preset))

    def on_switch_planner_done(self, future, preset) -> None:
        try:
            self.get_logger().info(f'Switch planner: {future.result().message}')
        except Exception as e:
            self.get_logger().warn(f'switch_planner failed: {e}')
        req = SetParamPreset.Request()
        req.preset = preset
        future = self.set_preset_client.call_async(req)
        future.add_done_callback(self.on_set_preset_done)

    def on_set_preset_done(self, future) -> None:
        try:
            self.get_logger().info(f'Set preset: {future.result().message}')
        except Exception as e:
            self.get_logger().warn(f'set_param_preset failed: {e}')
        self.rl_switch_called = False
        self.rl_state = S_START_EPISODE

    def do_map_switch(self, map_name: str) -> None:
        maps_dir = os.path.join(get_package_share_directory('simulation_launch'), 'maps')
        map_url = os.path.join(maps_dir, map_name, f'{map_name}.yaml')
        self.get_logger().info(f'Switching map to {map_name}')
        req = LoadMap.Request()
        req.map_url = map_url
        future = self.load_map_client.call_async(req)
        future.add_done_callback(lambda f: self.on_map_switch_done(f, map_name))

    def on_map_switch_done(self, future, map_name: str) -> None:
        try:
            res = future.result()
            if res.result == LoadMap.Response.RESULT_SUCCESS:
                self.get_logger().info(f'Map switched to {map_name}')
                self.rl_current_map = map_name
            else:
                self.get_logger().error(f'Map switch failed (result={res.result}), proceeding anyway')
                self.rl_current_map = map_name
        except Exception as e:
            self.get_logger().error(f'Map switch error: {e}')
        episode = self.rl_episodes[self.rl_episode_index]
        self.do_respawn(episode['start'])
        self.rl_settle_until = self.get_clock().now() + rclpy.duration.Duration(seconds=SETTLE_SECS)
        self.rl_state = S_SETTLING

    def do_respawn(self, start: dict) -> None:
        # start coordinates are in map frame; convert to odom frame
        ox = self.get_parameter('map_to_odom_x').value
        oy = self.get_parameter('map_to_odom_y').value
        pose = Pose2D()
        pose.x     = start['x'] - ox
        pose.y     = start['y'] - oy
        pose.theta = start.get('yaw', 0.0)
        self.set_pose_pub.publish(pose)
        self.get_logger().info(
            f'Respawn → map ({start["x"]:.2f}, {start["y"]:.2f}) = odom ({pose.x:.2f}, {pose.y:.2f})'
        )

    def on_episode_reset_done(self, future, episode: dict) -> None:
        try:
            self.get_logger().info(f'Metrics reset: {future.result().message}')
        except Exception as e:
            self.get_logger().warn(f'reset_metrics failed: {e}')
        self.rl_nav_status = None
        self.send_episode_goal(episode['goal'])

    def send_episode_goal(self, goal: dict) -> None:
        self.get_logger().info(f'Sending goal: {goal["goal_id"]}')

        # Publish goal so metrics_tracker can compute distance/heading error
        goal_pose = Pose()
        goal_pose.position.x = goal['x']
        goal_pose.position.y = goal['y']
        goal_pose.orientation.z = math.sin(goal['yaw'] / 2.0)
        goal_pose.orientation.w = math.cos(goal['yaw'] / 2.0)
        self.current_goal_pub.publish(goal_pose)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal['x']
        goal_msg.pose.pose.position.y = goal['y']
        goal_msg.pose.pose.orientation.z = math.sin(goal['yaw'] / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(goal['yaw'] / 2.0)
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.on_episode_goal_accepted)

    def on_episode_goal_accepted(self, future) -> None:
        try:
            goal_ptr: ClientGoalHandle = future.result()
        except Exception as e:
            self.get_logger().error(f'Goal send failed: {e}')
            self.rl_nav_status = 'FAILED'
            return
        if not goal_ptr.accepted:
            self.get_logger().warn('Episode goal rejected by Nav2')
            self.rl_nav_status = 'REJECTED'
            return
        self.get_logger().info('Episode goal accepted')
        result_future = goal_ptr.get_result_async()
        result_future.add_done_callback(self.on_episode_goal_done)

    def on_episode_goal_done(self, future) -> None:
        try:
            self.rl_nav_status = decode_nav2_status(future.result().status)
        except Exception as e:
            self.get_logger().error(f'Goal result error: {e}')
            self.rl_nav_status = 'ERROR'

    def on_episode_metrics_done(self, future) -> None:
        episode  = self.rl_episodes[self.rl_episode_index]
        nav_stat = self.rl_nav_status
        self.rl_nav_status = None
        try:
            res = future.result()
            m = res.metrics
            planner, preset = RUN_CONFIGS[self.rl_run_index]
            goal     = episode['goal']
            start    = episode['start']
            map_name = episode.get('map', self.get_parameter('default_map').value)

            m.planner      = planner
            m.preset       = preset
            m.map_name     = map_name
            m.goal_id      = goal['goal_id']
            m.goal_reached = (nav_stat == 'SUCCEEDED')
            m.start_x      = start['x']
            m.start_y      = start['y']
            m.start_yaw    = start.get('yaw', 0.0)
            m.goal_x       = goal['x']
            m.goal_y       = goal['y']
            m.goal_yaw     = goal['yaw']

            self.get_logger().info(f'Episode metrics: {res.message}')
            self.metrics_pub.publish(m)
        except Exception as e:
            self.get_logger().error(f'get_metrics failed: {e}')
        rl_mode = self.get_parameter('rl_mode').value
        if rl_mode:
            # Wrap episode index so training can run indefinitely
            self.rl_episode_index = (self.rl_episode_index + 1) % len(self.rl_episodes)
            self.rl_state = S_WAITING_FOR_RL
        else:
            self.rl_episode_index += 1
            self.rl_state = S_START_EPISODE

    def handle_start_episode(self, req, res):
        if self.rl_state != S_WAITING_FOR_RL:
            res.success = False
            res.message = f'Not waiting for RL trigger (state={self.rl_state})'
            return res
        if not self.rl_episodes:
            res.success = False
            res.message = 'No episodes loaded yet'
            return res
        self.get_logger().info(f'RL trigger: starting episode {self.rl_episode_index}')
        self.rl_state = S_START_EPISODE
        res.success = True
        res.message = f'Starting episode {self.rl_episode_index}'
        return res

    def check_nav2_active(self) -> None:
        if not self.nav2_state_client.service_is_ready():
            return
        future = self.nav2_state_client.call_async(GetState.Request())
        future.add_done_callback(self.process_nav2_state)

    def process_nav2_state(self, future: Future) -> None:
        try:
            self.nav2_active = (future.result().current_state.label == 'active')
        except Exception as e:
            self.get_logger().warn(f'Lifecycle check failed: {e}')

    def reset_action_goal_state(self) -> None:
        self.sendgoal_goal_ptr: ServerGoalHandle = None
        self.action_nav2_goal_ptr: ClientGoalHandle = None
        self.goal_id: str = None
        self.x_goal: float = None
        self.y_goal: float = None
        self.yaw_goal: float = None
        self.xy_tolerance: float = None
        self.yaw_tolerance: float = None
        self.action_nav2_status: str = None
        self.goal_transmitted: bool = False

    def manage_send_goal(self, goal_ptr: ServerGoalHandle) -> SendGoalToNav2.Result:
        request = goal_ptr.request
        self.get_logger().info(
            f"Received goal: id={request.goal_id}, x={request.x}, y={request.y}, yaw={request.yaw}"
        )
        if self.action_nav2_goal_ptr is not None:
            self.get_logger().warn(f"Canceling previous goal (id={self.goal_id})")
            self.action_nav2_goal_ptr.cancel_goal_async()
        self.sendgoal_goal_ptr = goal_ptr
        self.goal_id = request.goal_id or f"run_{round(time.time())}"
        self.x_goal = request.x
        self.y_goal = request.y
        self.yaw_goal = request.yaw
        self.xy_tolerance = request.xy_tolerance
        self.yaw_tolerance = request.yaw_tolerance
        self.action_nav2_status = None
        self.call_reset_metrics()
        while rclpy.ok() and self.sendgoal_goal_ptr is not None:
            time.sleep(0.05)
        final_result = copy.deepcopy(self.pending_result) if self.pending_result else SendGoalToNav2.Result()
        self.pending_result = None
        if self.sendgoal_success:
            goal_ptr.succeed(final_result)
        else:
            goal_ptr.canceled(final_result)
        return final_result

    def monitor_goal(self) -> None:
        if self.sendgoal_goal_ptr is None:
            return
        if not self.nav2_active:
            return
        if (self.action_nav2_goal_ptr is None) and (not self.goal_transmitted):
            self.transmit_action_goal(self.x_goal, self.y_goal, self.yaw_goal)
            self.goal_transmitted = True
            self.manual_nav2_override = False
            return
        goal_ptr = self.sendgoal_goal_ptr
        xy_error, yaw_error = self.compute_errors()
        if xy_error is not None and yaw_error is not None:
            feedback = SendGoalToNav2.Feedback()
            feedback.goal_id = self.goal_id
            feedback.curr_xy_error = xy_error
            feedback.curr_yaw_error = yaw_error
            goal_ptr.publish_feedback(feedback)
        result = SendGoalToNav2.Result()
        result.goal_id = self.goal_id
        result.nav2_status = self.action_nav2_status or "UNKNOWN"
        result.goal_reached = False
        result.final_xy_error = xy_error if xy_error is not None else float('inf')
        result.final_yaw_error = yaw_error if yaw_error is not None else float('inf')
        if goal_ptr.is_cancel_requested:
            self.action_nav2_goal_ptr.cancel_goal_async()
            self.pending_result = result
            self.sendgoal_success = False
            self.reset_action_goal_state()
            return
        if (
            xy_error is not None and yaw_error is not None and
            xy_error <= self.xy_tolerance and yaw_error <= self.yaw_tolerance
        ):
            if self.action_nav2_goal_ptr:
                self.action_nav2_goal_ptr.cancel_goal_async()
            result.goal_reached = True
            result.nav2_status = self.action_nav2_status or "UNKNOWN"
            self.pending_result = result
            self.sendgoal_success = True
            self.reset_action_goal_state()
            self.manual_nav2_override = True
            return
        nav2_status_snapshot = self.action_nav2_status
        if (not self.manual_nav2_override) and (nav2_status_snapshot is not None):
            result.nav2_status = nav2_status_snapshot
            self.pending_result = result
            self.sendgoal_success = True
            self.reset_action_goal_state()
            self.manual_nav2_override = False

    def transmit_action_goal(self, x: float, y: float, yaw: float) -> None:
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.on_action_goal_accepted)

    def on_action_goal_accepted(self, future: Future) -> None:
        goal_ptr: ClientGoalHandle = future.result()
        if not goal_ptr.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            return
        self.action_nav2_goal_ptr = goal_ptr
        goal_ptr.get_result_async().add_done_callback(self.on_action_nav2_result)

    def on_action_nav2_result(self, future: Future) -> None:
        self.action_nav2_status = decode_nav2_status(future.result().status)

    def compute_errors(self) -> Tuple[Optional[float], Optional[float]]:
        xy_error, yaw_error = None, None
        if (self.action_nav2_goal_ptr is None) or not self.goal_id:
            return xy_error, yaw_error
        try:
            curr_tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            curr_x = curr_tf.transform.translation.x
            curr_y = curr_tf.transform.translation.y
            curr_q = curr_tf.transform.rotation
            curr_yaw = math.atan2(
                2.0 * (curr_q.w * curr_q.z + curr_q.x * curr_q.y),
                1.0 - 2.0 * (curr_q.y * curr_q.y + curr_q.z * curr_q.z)
            )
            xy_error = math.sqrt((curr_x - self.x_goal)**2 + (curr_y - self.y_goal)**2)
            yaw_error = abs(((curr_yaw - self.yaw_goal) + math.pi) % (2 * math.pi) - math.pi)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
        return xy_error, yaw_error

    def call_reset_metrics(self) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('reset_metrics not available, skipping')
            return
        future = self.reset_client.call_async(Trigger.Request())
        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > 2.0:
                self.get_logger().warn('reset_metrics timed out')
                return
            time.sleep(0.01)
        self.get_logger().info(f'Metrics reset: {future.result().message}')


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRunner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
