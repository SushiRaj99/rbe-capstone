#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class LidarModel(Node):
    def __init__(self):
        super().__init__('lidar_model')

        self.occ = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.resolution = 0.05
        self.width = 0
        self.height = 0
        self.steps = None

        self.angle_min = -math.pi
        self.angle_max = math.pi
        self.angle_increment = math.radians(1.0)
        self.range_max = 10.0
        self.range_min = 0.05

        n_rays = int(round((self.angle_max - self.angle_min) / self.angle_increment)) + 1
        self.local_angles = np.linspace(self.angle_min, self.angle_max, n_rays)

        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to /map with transient-local QoS so we latch the current
        # map on connect AND get fresh messages whenever episode_runner calls
        # LoadMap. A one-shot GetMap service call at startup would leave the
        # lidar ray-tracing against the stale map forever after a switch.
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, '/map', self.map_cb, latched_qos)

        self.timer = self.create_timer(0.1, self.publish_scan)

    def map_cb(self, msg: OccupancyGrid) -> None:
        info = msg.info
        self.origin_x = info.origin.position.x
        self.origin_y = info.origin.position.y
        self.resolution = info.resolution
        self.width = info.width
        self.height = info.height
        self.occ = (
            np.asarray(msg.data, dtype=np.int8).reshape(self.height, self.width) > 50
        )
        # One step per cell — matches grid resolution, no benefit to oversampling.
        self.steps = np.arange(self.range_min, self.range_max + self.resolution, self.resolution)
        self.get_logger().info(
            f'Map received ({self.width}x{self.height} @ {self.resolution} m).'
        )

    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        x = t.transform.translation.x
        y = t.transform.translation.y
        yaw = quat_to_yaw(t.transform.rotation)
        return x, y, yaw

    def publish_scan(self):
        if self.occ is None:
            return
        pose = self.get_robot_pose()
        if pose is None:
            return
        x, y, yaw = pose

        global_angles = self.local_angles + yaw
        dx = np.cos(global_angles)[:, None]
        dy = np.sin(global_angles)[:, None]
        ss = self.steps[None, :]

        rx = x + ss * dx
        ry = y + ss * dy
        mx = ((rx - self.origin_x) / self.resolution).astype(np.int32)
        my = ((ry - self.origin_y) / self.resolution).astype(np.int32)

        in_bounds = (mx >= 0) & (mx < self.width) & (my >= 0) & (my < self.height)
        # Out-of-bounds cells act as occupied so rays terminate at the map edge.
        hit = np.ones_like(in_bounds, dtype=bool)
        hit[in_bounds] = self.occ[my[in_bounds], mx[in_bounds]]

        first_hit = hit.argmax(axis=1)
        ranges = self.steps[first_hit].astype(np.float32)
        ranges[~hit.any(axis=1)] = self.range_max

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.angle_min = float(self.angle_min)
        msg.angle_max = float(self.angle_max)
        msg.angle_increment = float(self.angle_increment)
        msg.range_min = float(self.range_min)
        msg.range_max = float(self.range_max)
        msg.ranges = ranges.tolist()
        self.scan_pub.publish(msg)


def main():
    rclpy.init()
    node = LidarModel()
    try:
        rclpy.spin(node)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
