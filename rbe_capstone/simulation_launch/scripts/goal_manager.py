#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.action.client import ClientGoalHandle
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from simulation_launch.action import SendGoalToNav2
from lifecycle_msgs.srv import GetState

import time, math, copy
from typing import Tuple, Optional


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# Helper function to map action_msgs/GoalStatus enum to associated string (defined in action_msgs/msg/GoalStatus.msg):
def decode_nav2_status(status_code: int) -> str:
    decoder = {
        0: "UNKNOWN",       # Goal has not been accepted yet, or its state is unknown
        1: "ACCEPTED",      # Goal was accepted, waiting to start executing
        2: "EXECUTING",     # Goal is currently being executed
        3: "CANCELING",     # Cancelation has been requested; waiting for confirmation
        4: "SUCCEEDED",     # Goal was completed successfully
        5: "CANCELED",      # Goal was successfully canceled
        6: "ABORTED",       # Goal was aborted (e.g. planner failure, obstacle, timeout)
    }
    return decoder.get(status_code, f"UNKNOWN_CODE_{status_code}")  # using get() to avoid KeyError for unexpected codes

# This node is designed to be able to send goal poses to Nav2 in 3 important use cases:
#   1) From the command line dynamically (for troubleshooting)
#   2) Programmatically through a gym environment (for RL training via Stable-Baselines3)
#   3) Programmatically through a launch file (for testing and evaluation) 
class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')
        self.cb_group = ReentrantCallbackGroup()  # allows timers to tick while manage_send_goal is blocking
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.client.wait_for_server()
        self.action_server = ActionServer(self, SendGoalToNav2, 'send_goal_to_nav2', self.manage_send_goal, callback_group=self.cb_group)
        self.reset_goal_state()
        self.pending_result = None
        self.sendgoal_success = False
        self.manual_nav2_override = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav2_state_client = self.create_client(GetState, '/bt_navigator/get_state')
        self.nav2_active = False
        self.monitor_timer = self.create_timer(0.1, self.monitor_goal, callback_group=self.cb_group)
        self.nav2_state_timer = self.create_timer(0.1, self.check_nav2_active, callback_group=self.cb_group)

    def reset_goal_state(self) -> None:
        self.sendgoal_goal_ptr: ServerGoalHandle = None
        self.nav2_goal_ptr: ClientGoalHandle = None
        self.goal_id: str = None
        self.x_goal: float = None
        self.y_goal: float = None
        self.yaw_goal: float = None
        self.xy_tolerance: float = None
        self.yaw_tolerance: float = None
        self.nav2_status: str = None
        self.goal_transmitted: bool = False

    def manage_send_goal(self, goal_ptr: ServerGoalHandle) -> SendGoalToNav2.Result:
        request = goal_ptr.request
        self.get_logger().info(
            f"Received goal: id={request.goal_id}, x={request.x}, y={request.y}, yaw={request.yaw}, xy_tolerance={request.xy_tolerance}, yaw_tolerance={request.yaw_tolerance}"
        )
        # Cancel any existing goal before sending a new one:
        if self.nav2_goal_ptr is not None:
            self.get_logger().warn(f"Canceling previous goal (id={self.goal_id})")
            self.nav2_goal_ptr.cancel_goal_async()
        # Store state:
        self.sendgoal_goal_ptr = goal_ptr
        self.goal_id = request.goal_id or f"run_{round(time.time())}"
        self.x_goal = request.x
        self.y_goal = request.y
        self.yaw_goal = request.yaw
        self.xy_tolerance = request.xy_tolerance
        self.yaw_tolerance = request.yaw_tolerance
        self.nav2_status = None
        # Block here until monitor_goal() settles the goal state (succeed/canceled) and clears sendgoal_goal_ptr.
        # This is safe because the ReentrantCallbackGroup + MultiThreadedExecutor allows the timers to keep ticking
        # concurrently. Without this block, the execute callback returns immediately, ROS2 auto-aborts the goal
        # handle, and any later call to goal_ptr.succeed() crashes with an invalid state transition.
        while rclpy.ok() and self.sendgoal_goal_ptr is not None:
            time.sleep(0.05)
        final_result = copy.deepcopy(self.pending_result) if (self.pending_result is not None) else SendGoalToNav2.Result() # using deepcopy() due to paranoia about mutability with the clearing of self.pending_result below
        #self.get_logger().info(f"[DEBUG] manage_send_goal returning: pending_result={self.pending_result}")   # uncomment for debugging result passing issues
        self.pending_result = None  # clear pending result for next goal
        if (self.sendgoal_success): 
            goal_ptr.succeed(final_result)
        else:
            goal_ptr.canceled(final_result)
        return final_result

    def monitor_goal(self) -> None:
        # Confirm that a valid request to send a goal to Nav2 was made:
        if self.sendgoal_goal_ptr is None:
            return
        # Confirm Nav2 is active:
        if not self.nav2_active:
            self.get_logger().info("Waiting for Nav2 to become active...")
            return
        # Ensure that the goal point has been sent to Nav2 once before proceeding:
        if (self.nav2_goal_ptr is None) and (not self.goal_transmitted):
            self.get_logger().info("Nav2 active, sending goal...")
            self.transmit_goal(self.x_goal, self.y_goal, self.yaw_goal)
            self.goal_transmitted = True
            self.manual_nav2_override = False
            return
        goal_ptr = self.sendgoal_goal_ptr
        xy_error, yaw_error = self.compute_errors()
        # Publish feedback:
        if xy_error is not None and yaw_error is not None:
            feedback = SendGoalToNav2.Feedback()
            feedback.goal_id = self.goal_id
            feedback.curr_xy_error = xy_error
            feedback.curr_yaw_error = yaw_error
            goal_ptr.publish_feedback(feedback)
        # Prepare result:
        result = SendGoalToNav2.Result()
        result.goal_id = self.goal_id
        result.nav2_status = self.nav2_status if self.nav2_status is not None else "UNKNOWN"
        result.goal_reached = False
        result.final_xy_error = xy_error if xy_error is not None else float('inf')
        result.final_yaw_error = yaw_error if yaw_error is not None else float('inf')
        # Cancel request
        if goal_ptr.is_cancel_requested:
            self.get_logger().info(f"Goal canceled (id={self.goal_id})")
            self.nav2_goal_ptr.cancel_goal_async()  # Attempt to cancel Nav2 goal as well to keep goal_manager and nav2 states consistent
            result.nav2_status = self.nav2_status if self.nav2_status is not None else "UNKNOWN"
            self.pending_result = result    # passes result to manage_send_goal() before unblocking it
            self.sendgoal_success = False
            self.reset_goal_state()         # clears sendgoal_goal_ptr, unblocking manage_send_goal()
            self.manual_nav2_override = False
            return
        # Success condition
        if (
            xy_error is not None and yaw_error is not None and
            xy_error <= self.xy_tolerance and yaw_error <= self.yaw_tolerance
        ):
            self.get_logger().info(f"Goal reached (id={self.goal_id})")
            if self.nav2_goal_ptr:
                self.nav2_goal_ptr.cancel_goal_async()  # Allows for early termination of nav2 process to speed up training
            result.goal_reached = True
            result.nav2_status = self.nav2_status if self.nav2_status else "UNKNOWN"
            self.pending_result = result    # passes result to manage_send_goal() before unblocking it
            self.sendgoal_success = True
            self.reset_goal_state()         # clears sendgoal_goal_ptr, unblocking manage_send_goal()
            self.manual_nav2_override = True
            return
        # Nav2 finished
        nav2_status_snapshot = self.nav2_status  # take snapshot of nav2 status to avoid race conditions
        if (not self.manual_nav2_override) and (nav2_status_snapshot is not None):
            self.get_logger().info(f"Nav2 finished first ({self.nav2_status})")
            result.nav2_status = nav2_status_snapshot
            self.pending_result = result    # passes result to manage_send_goal() before unblocking it
            self.sendgoal_success = True
            self.reset_goal_state()         # clears sendgoal_goal_ptr, unblocking manage_send_goal()
            self.manual_nav2_override = False
            return

    def transmit_goal(self, x: float, y: float, yaw: float) -> None:
        self.client.wait_for_server()
        goal_msg = NavigateToPose.Goal()
        # Convert 2D pose command (x, y, yaw) to 3D (x, y, z with quaternion orientation), then send it to Nav2:
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self.process_goal_response)
    
    def check_nav2_active(self) -> None:
        if not self.nav2_state_client.service_is_ready():
            return
        req = GetState.Request()
        future = self.nav2_state_client.call_async(req)
        future.add_done_callback(self.process_nav2_state)

    def process_nav2_state(self, future: Future) -> None:
        try:
            state = future.result().current_state.label
            #self.get_logger().info(f'[DEBUG] bt_navigator state: {state}') # uncomment for debugging lifecycle issues with Nav2
            self.nav2_active = (state == 'active')
        except Exception as e:
            self.get_logger().warn(f'Lifecycle check failed: {e}')

    def process_goal_response(self, future: Future) -> None:
        goal_ptr: ClientGoalHandle = future.result()
        if not goal_ptr.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            return
        self.get_logger().info("Nav2 goal accepted")
        self.nav2_goal_ptr = goal_ptr
        future_result = goal_ptr.get_result_async()
        future_result.add_done_callback(self.process_nav2_result)
    
    def process_nav2_result(self, future: Future) -> None:
        result: ClientGoalHandle = future.result()
        self.nav2_status = decode_nav2_status(result.status)

    def compute_errors(self) -> Tuple[Optional[float], Optional[float]]:
        xy_error, yaw_error = None, None
        if (self.nav2_goal_ptr is None) or (len(self.goal_id) == 0):
            return xy_error, yaw_error
        # Attempt to acquire transform and evaluate error relative to goal pose and tolerances:
        try:
            curr_tf = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            curr_x = curr_tf.transform.translation.x
            curr_y = curr_tf.transform.translation.y
            curr_yaw = quat_to_yaw(curr_tf.transform.rotation)
            xy_error = math.sqrt((curr_x - self.x_goal)**2 + (curr_y - self.y_goal)**2)
            yaw_error = abs(
                ((curr_yaw - self.yaw_goal) + math.pi) % (2*math.pi) - math.pi  # express error within range of -pi to pi
                )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
        return xy_error, yaw_error

def main(args=None):
    rclpy.init(args=args)
    node = GoalManager()
    # MultiThreadedExecutor is required so that the monitor_goal() and check_nav2_active() timers
    # can tick concurrently while manage_send_goal() is blocking in its spin-wait loop.
    # Without this, manage_send_goal() would deadlock waiting for monitor_goal() to unblock it.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
