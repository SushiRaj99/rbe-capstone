#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from tf2_ros import Buffer, TransformListener
import math
import numpy as np
import os


class LidarModel(Node):
    def __init__(self):
        super().__init__('lidar_model')

        self.map = None

        self.map_client = self.create_client(GetMap, '/map_server/map')
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for map service to activate...")
        self.map_req = GetMap.Request()
        self.future = self.map_client.call_async(self.map_req)
        self.future.add_done_callback(self.map_callback)

        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.1, self.publish_scan)

        # LiDAR parameters
        # TODO - eventually will want to make these configurable via a .yaml file
        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = math.radians(1.0)
        self.range_max = 10.0
        self.range_min = 0.05

    def map_callback(self, future):
        try:
            response = future.result()
            if response is not None:
                self.get_logger().info("Map received successfully!")
                self.map = response.map
            else:
                self.logger().warn("Map service call received empty response.")
        except Exception as e:
            self.get_logger().error(f"Map service call failed: {e!r}")

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y

            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            return x, y, yaw
        except:
            self.get_logger().error("failed to acquire transform from map->odom->base_link")
            return None

    def world_to_map(self, x, y):
        origin = self.map.info.origin.position
        res = self.map.info.resolution

        mx = int((x - origin.x) / res)
        my = int((y - origin.y) / res)
        return mx, my

    def is_occupied(self, mx, my):
        if mx < 0 or my < 0:
            return True
        if mx >= self.map.info.width or my >= self.map.info.height:
            return True

        idx = my * self.map.info.width + mx
        return self.map.data[idx] > 50  # TODO - using hard coded threshold for now but may want to revisit this

    def raycast(self, x, y, theta):
        step = self.map.info.resolution * 0.5
        dist = 0.0

        while dist < self.range_max:
            rx = x + dist * math.cos(theta)
            ry = y + dist * math.sin(theta)

            mx, my = self.world_to_map(rx, ry)

            if self.is_occupied(mx, my):
                return dist

            dist += step

        return self.range_max

    def publish_scan(self):
        if self.map is None:
            return
        stuff = GetMap.Request()

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.range_min = self.range_min
        msg.range_max = self.range_max

        ranges, intensities = [], []

        angle = self.angle_min
        while angle <= self.angle_max:
            r = self.raycast(x, y, yaw + angle)
            ranges.append(r)
            angle += self.angle_increment

        msg.ranges = ranges
        self.scan_pub.publish(msg)
        #self.get_logger().debug("ranges=" + str(ranges))


def main():
    rclpy.init()
    node = LidarModel()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()