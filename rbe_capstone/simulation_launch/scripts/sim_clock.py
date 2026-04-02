#!/usr/bin/env python3
import rclpy
import time
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


class SimClock(Node):
    def __init__(self):
        super().__init__('sim_clock', allow_undeclared_parameters=False)
        self.declare_parameter('speed_factor', 1.0)

        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.sim_time_ns = int(time.time() * 1e9)
        self.last_wall = time.monotonic()
        # Publish at 100 Hz — fast enough for Nav2's internal timers
        self.create_timer(0.01, self.publish_clock)

        speed = self.get_parameter('speed_factor').value
        self.get_logger().info(f'Sim clock running at {speed}x real time')

    def publish_clock(self):
        now = time.monotonic()
        dt_wall_ns = int((now - self.last_wall) * 1e9)
        self.last_wall = now

        speed = self.get_parameter('speed_factor').value
        self.sim_time_ns += int(dt_wall_ns * speed)

        msg = Clock()
        msg.clock.sec     = self.sim_time_ns // 1_000_000_000
        msg.clock.nanosec = self.sim_time_ns  % 1_000_000_000
        self.clock_pub.publish(msg)


def main():
    rclpy.init()
    node = SimClock()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
