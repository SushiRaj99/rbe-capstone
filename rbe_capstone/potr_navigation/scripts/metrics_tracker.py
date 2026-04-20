#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from potr_navigation.srv import GetMetrics
from potr_navigation.msg import EpisodeMetrics, StepMetrics


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class MetricsTracker(Node):
    def __init__(self):
        super().__init__('metrics_tracker')
        self.cb = ReentrantCallbackGroup()
        self.current_plan = []
        self.costmap = None          # /local_costmap/costmap  — odom frame, 3×3m rolling, 5Hz
        self.global_costmap = None   # /global_costmap/costmap — map frame, full map, 1Hz
        self.current_goal = None
        self.last_clearance = 0.0
        self.last_path_dev = 0.0
        self.last_collision = False
        self.reset_accumulators()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Odometry,      '/odom',                          self.odom_cb,     10, callback_group=self.cb)
        self.create_subscription(LaserScan,     '/scan',                          self.scan_cb,     10, callback_group=self.cb)
        self.create_subscription(Path,          '/plan',                          self.plan_cb,     10, callback_group=self.cb)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',         self.costmap_cb,        1, callback_group=self.cb)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',        self.global_costmap_cb, 1, callback_group=self.cb)
        self.create_subscription(Pose,          '/potr_navigation/current_goal',  self.goal_cb,     10, callback_group=self.cb)

        self.create_service(Trigger,    '/potr_navigation/reset_metrics', self.handle_reset, callback_group=self.cb)
        self.create_service(GetMetrics, '/potr_navigation/get_metrics',   self.handle_get,   callback_group=self.cb)

        self.step_pub = self.create_publisher(StepMetrics, '/potr_navigation/step_metrics', 10)

        self.get_logger().info('Metrics tracker ready')

    def reset_accumulators(self):
        self.start_time    = time.monotonic()
        self.total_dist    = 0.0
        self.min_clearance = float('inf')
        self.path_devs     = []
        self.last_pos      = None
        self.collision_count = 0
        # Drop stale plan — path_deviation should only be measured
        # against a plan published for the current episode's goal.
        self.current_plan  = []
        self.last_path_dev = 0.0

        self.spd_n, self.spd_mean, self.spd_M2 = 0, 0.0, 0.0
        self.hdg_n, self.hdg_mean, self.hdg_M2 = 0, 0.0, 0.0

    def odom_cb(self, msg):
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y

        yaw = quat_to_yaw(msg.pose.pose.orientation)

        # Plan and goal live in the map frame, odom does not — look up
        # base_link in map so path_deviation / distance_to_goal are consistent.
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
        except Exception:
            x, y = odom_x, odom_y

        if self.last_pos is not None:
            dx = odom_x - self.last_pos[0]
            dy = odom_y - self.last_pos[1]
            self.total_dist += math.sqrt(dx * dx + dy * dy)
        self.last_pos = (odom_x, odom_y)

        collision = False
        if self.costmap is not None:
            # /local_costmap/costmap is in odom frame — use odom coordinates
            # for cell lookup regardless of which frame we resolved for the plan.
            if get_costmap_cost(odom_x, odom_y, self.costmap) >= 100:
                self.collision_count += 1
                collision = True
        self.last_collision = collision

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        speed = math.sqrt(vx * vx + vy * vy)
        self.spd_n, self.spd_mean, self.spd_M2 = welford(
            self.spd_n, self.spd_mean, self.spd_M2, speed
        )

        wz = abs(msg.twist.twist.angular.z)
        self.hdg_n, self.hdg_mean, self.hdg_M2 = welford(
            self.hdg_n, self.hdg_mean, self.hdg_M2, wz
        )

        path_dev = 0.0
        if len(self.current_plan) >= 2:
            dev = point_to_path_dist(x, y, self.current_plan)
            if dev is not None:
                self.path_devs.append(dev)
                path_dev = dev
        self.last_path_dev = path_dev

        dist_to_goal = 0.0
        heading_err = 0.0
        if self.current_goal is not None:
            gx = self.current_goal.position.x
            gy = self.current_goal.position.y
            dist_to_goal = math.sqrt((gx - x) ** 2 + (gy - y) ** 2)
            bearing = math.atan2(gy - y, gx - x)
            heading_err = ((bearing - yaw) + math.pi) % (2 * math.pi) - math.pi

        cost_near, cost_mid, cost_far = self.compute_path_costs(x, y)

        s = StepMetrics()
        s.distance_to_goal      = dist_to_goal
        s.heading_error_to_goal = heading_err
        s.linear_velocity       = speed
        s.angular_velocity      = wz
        s.clearance             = self.last_clearance
        s.path_deviation        = path_dev
        s.collision             = collision
        if self.current_goal is not None:
            s.goal_x_map = self.current_goal.position.x
            s.goal_y_map = self.current_goal.position.y
        else:
            s.goal_x_map = 0.0
            s.goal_y_map = 0.0
        s.path_cost_near = cost_near
        s.path_cost_mid  = cost_mid
        s.path_cost_far  = cost_far
        self.step_pub.publish(s)

    def compute_path_costs(self, x_map, y_map):
        """Walk the global plan forward from the nearest plan-point to the
        robot, sampling the *global* costmap in three arc-length windows
        ahead: [0, 1 m], [1, 3 m], [3, 6 m]. Returns each window's max cost
        so the policy sees the worst obstacle on the upcoming planned path.

        Global costmap is in the map frame — same frame as the plan — so we
        can look each plan point up directly with no offset. The local
        costmap's 3x3m rolling window would silently return 0 for anything
        past ~1.5m, which is why we don't use it here.
        """
        if self.global_costmap is None or len(self.current_plan) < 2:
            return 0.0, 0.0, 0.0

        closest_i = 0
        closest_d2 = float('inf')
        for i, (px, py) in enumerate(self.current_plan):
            d2 = (px - x_map) ** 2 + (py - y_map) ** 2
            if d2 < closest_d2:
                closest_d2 = d2
                closest_i = i

        bins = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0))
        far_horizon = bins[-1][1]
        maxes = [0, 0, 0]
        acc = 0.0
        prev_px, prev_py = self.current_plan[closest_i]
        for (px, py) in self.current_plan[closest_i + 1:]:
            acc += math.hypot(px - prev_px, py - prev_py)
            if acc > far_horizon:
                break
            cost = get_costmap_cost(px, py, self.global_costmap)
            for b, (lo, hi) in enumerate(bins):
                if lo <= acc <= hi and cost > maxes[b]:
                    maxes[b] = cost
            prev_px, prev_py = px, py

        return float(maxes[0]), float(maxes[1]), float(maxes[2])

    def goal_cb(self, msg):
        self.current_goal = msg

    def scan_cb(self, msg):
        valid = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        if valid:
            current = min(valid)
            self.min_clearance = min(self.min_clearance, current)
            self.last_clearance = current

    def plan_cb(self, msg):
        self.current_plan = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]

    def costmap_cb(self, msg):
        self.costmap = msg

    def global_costmap_cb(self, msg):
        self.global_costmap = msg

    def handle_reset(self, req, res):
        self.reset_accumulators()
        res.success = True
        res.message = 'Metrics reset'
        self.get_logger().info('Metrics reset')
        return res

    def handle_get(self, req, res):
        m = EpisodeMetrics()
        m.total_time          = time.monotonic() - self.start_time
        m.total_distance      = self.total_dist
        m.min_clearance       = self.min_clearance if math.isfinite(self.min_clearance) else 0.0
        m.mean_path_deviation = (sum(self.path_devs) / len(self.path_devs)) if self.path_devs else 0.0
        m.max_path_deviation  = max(self.path_devs) if self.path_devs else 0.0
        m.velocity_variance   = (self.spd_M2 / (self.spd_n - 1)) if self.spd_n > 1 else 0.0
        m.heading_variance    = (self.hdg_M2 / (self.hdg_n - 1)) if self.hdg_n > 1 else 0.0
        m.collision_count     = self.collision_count

        res.metrics = m
        res.success = True
        res.message = (
            f'time={m.total_time:.1f}s  dist={m.total_distance:.2f}m  '
            f'clearance={m.min_clearance:.2f}m  dev={m.mean_path_deviation:.3f}m  '
            f'vel_var={m.velocity_variance:.4f}  hdg_var={m.heading_variance:.4f}  '
            f'collisions={m.collision_count}'
        )
        self.get_logger().info(f'Metrics: {res.message}')
        return res


def welford(n, mean, M2, x):
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)
    return n, mean, M2


def get_costmap_cost(px, py, costmap):
    mx = int((px - costmap.info.origin.position.x) / costmap.info.resolution)
    my = int((py - costmap.info.origin.position.y) / costmap.info.resolution)
    if mx < 0 or my < 0 or mx >= costmap.info.width or my >= costmap.info.height:
        return 0
    return costmap.data[my * costmap.info.width + mx]


def point_to_path_dist(px, py, path):
    min_dist = float('inf')
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        seg_sq = dx * dx + dy * dy
        if seg_sq < 1e-9:
            d = math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
            cx, cy = ax + t * dx, ay + t * dy
            d = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if d < min_dist:
            min_dist = d
    return min_dist if math.isfinite(min_dist) else None


def main(args=None):
    rclpy.init(args=args)
    node = MetricsTracker()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
