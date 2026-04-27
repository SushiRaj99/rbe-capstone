#!/usr/bin/env python3
import json
import os
import time
import yaml
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String
from potr_navigation.srv import SwitchPlanner, SetParamPreset, SetRawParams

VALID_PLANNERS = ('DWB',)
VALID_PRESETS = (1, 2)

# DWB is registered under the "FollowPath" plugin name in nav2_params.yaml so the
# stock Nav2 BT (which calls FollowPath by name) actually routes to it.
PLUGIN_NS = {'DWB': 'FollowPath'}

SHARED_PARAM_MAP = {
    'DWB': {
        'max_linear_vel': 'FollowPath.max_vel_x',
        'min_linear_vel': 'FollowPath.min_vel_x',
        'max_angular_vel': 'FollowPath.max_vel_theta',
        'linear_accel': 'FollowPath.acc_lim_x',
        'angular_accel': 'FollowPath.acc_lim_theta',
        'goal_align_scale': 'FollowPath.GoalAlign.scale',
        'path_align_scale': 'FollowPath.PathAlign.scale',
        'goal_dist_scale': 'FollowPath.GoalDist.scale',
        'path_dist_scale': 'FollowPath.PathDist.scale',
        'obstacle_scale': 'FollowPath.BaseObstacle.scale',
    },
}


class PlannerController(Node):
    def __init__(self):
        super().__init__('planner_controller')

        self.planner = 'DWB'
        self.preset = 1
        self.current_max_lin = 0.8
        self.current_max_ang = 1.2

        pkg = get_package_share_directory('potr_navigation')
        self.shared_yaml = os.path.join(pkg, 'config', 'shared_params.yaml')
        self.dwb_yaml = os.path.join(pkg, 'config', 'dwb_params.yaml')

        self.cb = ReentrantCallbackGroup()

        self.set_params_client = self.create_client(SetParameters, '/controller_server/set_parameters', callback_group=self.cb)
        self.smoother_params_client = self.create_client(SetParameters, '/velocity_smoother/set_parameters', callback_group=self.cb)

        self.latest_shared_values = {}
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.params_pub = self.create_publisher(String, '/potr_navigation/current_planner_params', latched_qos)
        self.publish_params_snapshot()

        self.create_service(SwitchPlanner, '/potr_navigation/switch_planner', self.handle_switch_planner, callback_group=self.cb)
        self.create_service(SetParamPreset, '/potr_navigation/set_param_preset', self.handle_set_preset, callback_group=self.cb)
        self.create_service(SetRawParams, '/potr_navigation/set_raw_params', self.handle_set_raw_params, callback_group=self.cb)

        self.get_logger().info(f'Planner controller ready (planner={self.planner}, preset={self.preset})')

    def handle_switch_planner(self, req, res):
        name = req.planner_name.upper().strip()
        if name not in VALID_PLANNERS:
            res.success = False
            res.message = f"Unknown planner '{name}'. Valid: {VALID_PLANNERS}"
            return res
        self.planner = name
        res.success, res.message = self.apply_params()
        return res

    def handle_set_preset(self, req, res):
        if req.preset not in VALID_PRESETS:
            res.success = False
            res.message = f"Invalid preset '{req.preset}'. Valid: {VALID_PRESETS}"
            return res
        self.preset = req.preset
        res.success, res.message = self.apply_params()
        return res

    def handle_set_raw_params(self, req, res):
        if len(req.names) != len(req.values):
            res.success = False
            res.message = 'names and values arrays must have equal length'
            return res
        params = {}
        shared_updates = {}
        for name, value in zip(req.names, req.values):
            ros_name = SHARED_PARAM_MAP[self.planner].get(name)
            if ros_name is None:
                res.success = False
                res.message = f"Unknown param '{name}' for planner {self.planner}"
                return res
            params[ros_name] = float(value)
            shared_updates[name] = float(value)
        summary = '  '.join(f'{n}={v:.3f}' for n, v in zip(req.names, req.values))
        self.get_logger().info(f'set_raw_params ({self.planner}): {summary}')
        ok, msg = self.send_params(params)
        if not ok:
            res.success = ok
            res.message = msg
            return res

        smoother_dirty = False
        if 'max_linear_vel' in shared_updates:
            self.current_max_lin = shared_updates['max_linear_vel']
            smoother_dirty = True
        if 'max_angular_vel' in shared_updates:
            self.current_max_ang = shared_updates['max_angular_vel']
            smoother_dirty = True
        if smoother_dirty:
            smoother_params = {
                'max_velocity': [self.current_max_lin, 0.0, self.current_max_ang],
                'min_velocity': [-self.current_max_lin, 0.0, -self.current_max_ang],
            }
            ok, msg = self.send_params(smoother_params, client=self.smoother_params_client)
            if not ok:
                res.success = False
                res.message = f'Smoother update failed: {msg}'
                return res

        self.latest_shared_values.update(shared_updates)
        self.publish_params_snapshot()
        res.success = True
        res.message = 'OK'
        return res

    def apply_params(self):
        planner = self.planner
        preset_key = f'preset_{self.preset}'
        plugin_ns = PLUGIN_NS[planner]
        params = {}
        shared_updates = {}
        reverse_shared = {v: k for k, v in SHARED_PARAM_MAP[planner].items()}

        try:
            with open(self.shared_yaml) as f:
                shared = yaml.safe_load(f)[preset_key]
            for key, val in shared.items():
                ros_name = SHARED_PARAM_MAP[planner].get(key)
                if ros_name:
                    params[ros_name] = val
                    shared_updates[key] = val
        except Exception as e:
            return False, f'Failed to load shared params: {e}'

        try:
            with open(self.dwb_yaml) as f:
                planner_params = yaml.safe_load(f)[preset_key]['controller_server']['ros__parameters'][plugin_ns]
            for k, v in planner_params.items():
                if k == 'critics':
                    continue  # live critics update crashes the controller
                ros_name = f'{plugin_ns}.{k}'
                params[ros_name] = v
                if ros_name in reverse_shared:
                    shared_updates[reverse_shared[ros_name]] = v
        except Exception as e:
            return False, f'Failed to load {planner} params: {e}'

        ok, msg = self.send_params(params)
        if not ok:
            return False, msg

        try:
            max_lin = shared.get('max_linear_vel', 0.5)
            max_ang = shared.get('max_angular_vel', 2.0)
            self.current_max_lin = max_lin
            self.current_max_ang = max_ang
            smoother_params = {
                'max_velocity': [max_lin, 0.0, max_ang],
                'min_velocity': [-max_lin, 0.0, -max_ang],
            }
            ok, msg = self.send_params(smoother_params, client=self.smoother_params_client)
            if not ok:
                return False, f'Smoother update failed: {msg}'
        except Exception as e:
            return False, f'Failed to update smoother params: {e}'

        self.latest_shared_values = shared_updates
        self.publish_params_snapshot()
        return True, 'OK'

    def send_params(self, params, client=None):
        if client is None:
            client = self.set_params_client
        if not client.wait_for_service(timeout_sec=5.0):
            return False, f'{client.srv_name} not available'

        ros_params = []
        for name, value in params.items():
            p = Parameter()
            p.name = name
            p.value = self.make_param_value(value)
            ros_params.append(p)

        req = SetParameters.Request()
        req.parameters = ros_params
        future = client.call_async(req)

        start = time.monotonic()
        while not future.done():
            if time.monotonic() - start > 5.0:
                return False, 'set_parameters timed out'
            time.sleep(0.01)

        if future.result() is None:
            return False, 'set_parameters returned no result'

        failures = [r.reason for r in future.result().results if not r.successful]
        if failures:
            return False, f'Parameters failed: {failures}'

        self.get_logger().info(f'Applied {len(ros_params)} params ({self.planner}, preset {self.preset})')
        return True, 'OK'

    def publish_params_snapshot(self):
        payload = json.dumps({
            'planner': self.planner,
            'preset': self.preset,
            'values': self.latest_shared_values,
        })
        msg = String()
        msg.data = payload
        self.params_pub.publish(msg)

    def make_param_value(self, value):
        pv = ParameterValue()
        if isinstance(value, bool):
            pv.type = ParameterType.PARAMETER_BOOL
            pv.bool_value = value
        elif isinstance(value, int):
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = value
        elif isinstance(value, float):
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = value
        elif isinstance(value, str):
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = value
        elif isinstance(value, list) and all(isinstance(v, float) for v in value):
            pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            pv.double_array_value = value
        else:
            pv.type = ParameterType.PARAMETER_STRING
            pv.string_value = str(value)
        return pv


def main(args=None):
    rclpy.init(args=args)
    node = PlannerController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
