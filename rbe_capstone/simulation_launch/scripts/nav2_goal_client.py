#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, ParameterValue
from simulation_launch.action import SendGoalToNav2

import math, json

class SendGoalToNav2Client(Node):
    def __init__(self):
        super().__init__('nav2_goal_client')
        self.cb_group = ReentrantCallbackGroup()
        self.client = ActionClient(self, SendGoalToNav2, 'send_goal_to_nav2', callback_group=self.cb_group)
        # Since we want to declare initial state of parameters as 'None' to avoid
        # queueing incomplete goal point definitions, the parameter descriptors will 
        # be used to avoid the deprecation warnings: 
        self.declare_parameter('goal_id', "")
        self.declare_parameter('x', float('nan'))
        self.declare_parameter('y', float('nan'))
        self.declare_parameter('yaw', float('nan'))
        self.declare_parameter('xy_tolerance', float('nan'))
        self.declare_parameter('yaw_tolerance', float('nan'))
        self.declare_parameter('goal_queue', "")
        # Set up a first-in-first-out (FIFO) queue for triggering send_goal_to_nav2 actions: 
        self.goal_FIFO = []
        self.main_timer = self.create_timer(0.5, self.main_loop, callback_group=self.cb_group)
        self.goal_active = False
        self.server_ready = False
    
    def main_loop(self) -> None:
        # Don't do anything if rclpy is down (e.g. during shutdown)
        if not rclpy.ok():
            return
        # Wait for action server to initialize (one-time):
        if not self.server_ready:
            if not self.client.wait_for_server(timeout_sec=0.0):    # need to set timeout to zero to prevent blocking other initialization activities
                self.get_logger().info('Waiting for send_goal_to_nav2 server...')
                return
            self.get_logger().info("Action server for send_goal_to_nav2 ready!")
            self.server_ready = True
        # Populate and Unload FIFO:
        self.populate_queue()
        self.unload_queue()

    def populate_queue(self) -> None:
        param_x = self.get_parameter('x').value
        param_y = self.get_parameter('y').value
        param_yaw = self.get_parameter('yaw').value
        param_xy_tolerance = self.get_parameter('xy_tolerance').value
        param_yaw_tolerance = self.get_parameter('yaw_tolerance').value
        param_goal_id = self.get_parameter('goal_id').value
        param_goal_q_str = self.get_parameter('goal_queue').value
        param_goal_q = []
        if len(param_goal_q_str.strip()) > 0:
            try:
                param_goal_q = json.loads(param_goal_q_str)
            except json.JSONDecodeError as e:
                self.get_logger().error(f"Failed to parse goal_queue JSON: {e}")
        added_singlept, added_waypts = False, False
        # Prioritize adding single goal point definitions to the FIFO first:
        if (not math.isnan(param_x)) and (not math.isnan(param_y)) and (not math.isnan(param_yaw)) and \
            (not math.isnan(param_xy_tolerance)) and (not math.isnan(param_yaw_tolerance)) and (len(param_goal_id.strip()) == 0):
            new_goal_pt = {
                'goal_id': param_goal_id,
                'x': param_x,
                'y': param_y,
                'yaw': param_yaw,
                'xy_tolerance': param_xy_tolerance,
                'yaw_tolerance': param_yaw_tolerance
            }
            self.goal_FIFO.append(new_goal_pt)
            self.get_logger().info(f"Added goal point: {new_goal_pt}")
            added_singlept = True
        # After any single points, add waypoints to the FIFO:
        for waypoint in param_goal_q:
            if (isinstance(waypoint, dict)) and (all(
                [this_param in list(waypoint.keys()) for this_param in ['goal_id', 'x', 'y', 'yaw', 'xy_tolerance', 'yaw_tolerance']]
                )):
                self.goal_FIFO.append(waypoint)
                self.get_logger().info(f"Added waypoint: {waypoint}")
            else:
                self.get_logger().warn(f"Invalid waypoint definition: {waypoint}")
            added_waypts = True
        # Finally, reset parameters after points have been added to the FIFO to prevent duplicates:
        if (added_singlept):
            self.set_parameters([
                Parameter('goal_id', value=""),
                Parameter('x', value=float('nan')),
                Parameter('y', value=float('nan')),
                Parameter('yaw', value=float('nan')),
                Parameter('xy_tolerance', value=float('nan')),
                Parameter('yaw_tolerance', value=float('nan')),
            ])
        if (added_waypts):
            self.set_parameters([Parameter('goal_queue', value="")])

    def unload_queue(self) -> None:
        # Make sure there are entries in the queue and that there are no other active goal points 
        # before attempting to unload the FIFO:
        if (self.goal_active) or (len(self.goal_FIFO) <= 0):
            return
        # Unload the FIFO one task at a time (no need for while loop here since it's operating on a timer):
        goal_task = self.goal_FIFO.pop(0)
        goal_msg = SendGoalToNav2.Goal()
        goal_msg.goal_id = goal_task['goal_id']
        goal_msg.x = goal_task['x']
        goal_msg.y = goal_task['y']
        goal_msg.yaw = goal_task['yaw']
        goal_msg.xy_tolerance = goal_task['xy_tolerance']
        goal_msg.yaw_tolerance = goal_task['yaw_tolerance']
        self.get_logger().info(f"Requesting goal: {goal_msg}")
        try:
            future = self.client.send_goal_async(goal_msg)
            future.add_done_callback(self.process_goalmanager_response)
            self.goal_active = True
        except Exception as e:
            self.get_logger().error(f"Failed to send goal: {e}")
            self.goal_active = False
    
    def process_goalmanager_response(self, future: Future) -> None:
        # Need to ensure early termination within this callback to ensure thread safety 
        # during initialization (no blocking):
        if not rclpy.ok() or self.context is None:
            return
        try:
            goal_ptr: ClientGoalHandle = future.result()
        except Exception as e:
            self.get_logger().error(f"SendGoalToNav2 response failed: {e}")
            self.goal_active = False
            return
        # If the goal handle was acquired check the status of the request:
        if not goal_ptr.accepted:
            self.get_logger().warn("SendGoalToNav2 request rejected")
            self.goal_active = False
            return
        self.get_logger().info("SendGoalToNav2 request accepted")
        # Attempt to configure the postprocessing for the SendGoalToNav2 server:
        try:
            future_result = goal_ptr.get_result_async()
            future_result.add_done_callback(self.process_goalmanager_result)
        except Exception as e:
            self.get_logger().error(f"Failed to configure SendGoalToNav2 post processing: {e}")
            self.goal_active = False

    def process_goalmanager_result(self, future: Future) -> None:
        # Need to ensure early termination within this callback to ensure thread safety 
        # during initialization (no blocking):
        if not rclpy.ok() or self.context is None:
            return
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"Failed to acquire result from SendGoalToNav2: {e}")
            self.goal_active = False
            return
        # Log success and reset active goal indicator:
        self.get_logger().info(
            f"Goal finished: GoalManager (success={result.goal_reached}), Nav2 (status={result.nav2_status})"
        )
        self.goal_active = False

    def destroy_node(self) -> None:
        # Need a 'destructor' to cleanly shut things down with the MultiThreadedExecutor and callback group:
        self.get_logger().info("Shutting down cleanly...")
        self.goal_active = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SendGoalToNav2Client()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()