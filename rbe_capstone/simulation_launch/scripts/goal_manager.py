#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.action.client import ClientGoalHandle
from rclpy.action.server import ServerGoalHandle
from rclpy.task import Future
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
#from tf_transformations import quaternion_from_euler
from tf2_ros import Buffer, TransformListener
from simulation_launch.action import SendGoalToNav2

import time, math
import asyncio
from typing import Tuple, Optional


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
    return decoder.get(status_code, f"UNKNOWN_CODE_{status_code}")  # using get() to avoids KeyError for unexpected codes

# This node is designed to be able to send goal poses to Nav2 in 3 important use cases:
#   1) From the command line dynamically (for troubleshooting)
#   2) Programmatically through a gym environment (for RL training via Stable-Baselines3)
#   3) Programmatically through a launch file (for testing and evaluation) 
class GoalManager(Node):
    def __init__(self):
        super().__init__('goal_manager')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.client.wait_for_server()
        self.action_server = ActionServer(self, SendGoalToNav2, 'send_goal_to_nav2', self.manage_send_goal)
        self.reset_goal_state()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

    async def manage_send_goal(self, goal_ptr: ServerGoalHandle) -> SendGoalToNav2.Result:
        request = goal_ptr.request
        self.get_logger().info(
            f"Received goal: id={request.goal_id}, x={request.x}, y={request.y}, yaw={request.yaw}, xy_tolerance={request.xy_tolerance}, yaw_tolerance={request.yaw_tolerance}"
        )
        # Cancel any existing goal before sending a new one:
        if self.nav2_goal_ptr != None:
            self.get_logger().warn(f"Cancelling previous goal (id={self.goal_id})")
            self.nav2_goal_ptr.cancel_goal_async()
        # Send goal to Nav2:
        self.send_goal(request.x, request.y, request.yaw)
        self.sendgoal_goal_ptr = goal_ptr
        self.goal_id = request.goal_id
        if (len(self.goal_id.strip()) == 0): self.goal_id = "run_" + str(round(time.time()))
        self.x_goal = request.x
        self.y_goal = request.y
        self.yaw_goal = request.yaw
        self.xy_tolerance = request.xy_tolerance
        self.yaw_tolerance = request.yaw_tolerance
        # Continually monitor progress:
        while rclpy.ok():
            # Compute errors and provide feedback:
            xy_error, yaw_error = self.compute_errors()
            if (xy_error != None) and (yaw_error != None):
                feedback = SendGoalToNav2.Feedback()
                feedback.goal_id = self.goal_id
                feedback.curr_xy_error = xy_error
                feedback.curr_yaw_error = yaw_error
                goal_ptr.publish_feedback(feedback)
            # Prepare a result and check for termination conditions:
            result = SendGoalToNav2.Result()
            result.goal_id = self.goal_id
            result.nav2_status = self.nav2_status if self.nav2_status != None else "UNKNOWN"  # guard: None is invalid for a string field
            result.goal_reached = False
            result.final_xy_error = xy_error if xy_error != None else float('inf')
            result.final_yaw_error = yaw_error if yaw_error != None else float('inf')
            if goal_ptr.is_cancel_requested:
                self.get_logger().info(f"Goal canceled (id={self.goal_id})")
                goal_ptr.canceled()
                return result
            elif ((self.xy_tolerance != None) and (self.yaw_tolerance != None)) and ((xy_error <= self.xy_tolerance) and (yaw_error <= self.yaw_tolerance)):
                self.get_logger().info(f"Goal reached (id={self.goal_id})")
                if self.nav2_goal_ptr:
                    self.nav2_goal_ptr.cancel_goal_async()  # Allows for early termination of nav2 process to speed up training
                result.goal_reached = True
                goal_ptr.succeed()
                return result
            elif (self.nav2_status != None):
                self.get_logger().info(f"Nav2 finished first (status={self.nav2_status})")
                goal_ptr.succeed()
                return result
            await asyncio.sleep(0.1)

    def send_goal(self, x: float, y: float, yaw: float) -> None:
        self.client.wait_for_server()
        goal_msg = NavigateToPose.Goal()
        # Convert 2D pose command (x, y, yaw) to 3D for Nav2 (x, y, z with quaternion orientation):
        #this_quat_list = quaternion_from_euler(0.0, 0.0, yaw)
        # Populate and send message:
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.x = 0.0 #this_quat_list[0]
        goal_msg.pose.pose.orientation.y = 0.0 #this_quat_list[1]
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0) #this_quat_list[2]
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0) #this_quat_list[3]
        future = self.client.send_goal_async(goal_msg)
        future.add_done_callback(self.process_goal_response)
    
    def process_goal_response(self, future: Future) -> None:
        goal_ptr: ClientGoalHandle = future.result()
        if not goal_ptr.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            return
        self.get_logger().info("Nav2 goal accepted")
        self.nav2_goal_ptr = goal_ptr
        self.nav2_status = decode_nav2_status(0)
        future_result = goal_ptr.get_result_async()
        future_result.add_done_callback(self.process_nav2_result)
    
    def process_nav2_result(self, future: Future) -> None:
        result: ClientGoalHandle = future.result()
        self.nav2_status = decode_nav2_status(result.status)

    def compute_errors(self) -> Tuple[Optional[float], Optional[float]]:
        xy_error, yaw_error = None, None
        if (self.nav2_goal_ptr == None) or (len(self.goal_id) == 0):
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
            curr_q = curr_tf.transform.rotation
            curr_yaw = math.atan2(
                2.0 * (curr_q.w * curr_q.z + curr_q.x * curr_q.y),
                1.0 - 2.0 * (curr_q.y * curr_q.y + curr_q.z * curr_q.z)
            )
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
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
