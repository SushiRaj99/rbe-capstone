#!/usr/bin/env python3
import time
import yaml
import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from ament_index_python.packages import get_package_share_directory
from potr_navigation.srv import SwitchPlanner, SetParamPreset

VALID_PLANNERS = ('DWB', 'MPPI')
VALID_PRESETS = (1, 2)

PLUGIN_NS = {'DWB': 'FollowPathDWB', 'MPPI': 'FollowPath'}

SHARED_PARAM_MAP = {
    'DWB': {
        'max_linear_vel':  'FollowPathDWB.max_vel_x',
        'min_linear_vel':  'FollowPathDWB.min_vel_x',
        'max_angular_vel': 'FollowPathDWB.max_vel_theta',
        'linear_accel':    'FollowPathDWB.acc_lim_x',
        'angular_accel':   'FollowPathDWB.acc_lim_theta',
    },
    'MPPI': {
        'max_linear_vel':  'FollowPath.vx_max',
        'min_linear_vel':  'FollowPath.vx_min',
        'max_angular_vel': 'FollowPath.wz_max',
        'linear_accel':    'FollowPath.ax_max',
        'angular_accel':   'FollowPath.az_max',
    },
}


class PlannerController(Node):
    def __init__(self):
        super().__init__('planner_controller')

        self.planner = 'MPPI'
        self.preset  = 1

        pkg = get_package_share_directory('potr_navigation')
        self.shared_yaml = os.path.join(pkg, 'config', 'shared_params.yaml')
        self.dwb_yaml    = os.path.join(pkg, 'config', 'dwb_params.yaml')
        self.mppi_yaml   = os.path.join(pkg, 'config', 'mppi_params.yaml')

        self.cb = ReentrantCallbackGroup()

        self.set_params_client = self.create_client(
            SetParameters, '/controller_server/set_parameters',
            callback_group=self.cb,
        )
        self.smoother_params_client = self.create_client(
            SetParameters, '/velocity_smoother/set_parameters',
            callback_group=self.cb,
        )
        self.create_service(
            SwitchPlanner, '/potr_navigation/switch_planner',
            self.handle_switch_planner, callback_group=self.cb,
        )
        self.create_service(
            SetParamPreset, '/potr_navigation/set_param_preset',
            self.handle_set_preset, callback_group=self.cb,
        )

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

    def apply_params(self):
        planner = self.planner
        preset_key = f'preset_{self.preset}'
        plugin_ns = PLUGIN_NS[planner]
        params = {}

        try:
            shared = yaml.safe_load(open(self.shared_yaml))[preset_key]
            for key, val in shared.items():
                ros_name = SHARED_PARAM_MAP[planner].get(key)
                if ros_name:
                    params[ros_name] = val
        except Exception as e:
            return False, f'Failed to load shared params: {e}'

        try:
            planner_yaml = self.dwb_yaml if planner == 'DWB' else self.mppi_yaml
            planner_params = (
                yaml.safe_load(open(planner_yaml))
                [preset_key]['controller_server']['ros__parameters'][plugin_ns]
            )
            for k, v in planner_params.items():
                if k == 'critics':
                    # setting critics on the fly seems to make things go kerplut
                    continue
                else:
                    params[f'{plugin_ns}.{k}'] = v
        except Exception as e:
            return False, f'Failed to load {planner} params: {e}'

        ok, msg = self.send_params(params)
        if not ok:
            return False, msg

        try:
            shared = yaml.safe_load(open(self.shared_yaml))[preset_key]
            max_lin = shared.get('max_linear_vel', 0.5)
            max_ang = shared.get('max_angular_vel', 2.0)
            smoother_params = {
                'max_velocity': [max_lin, 0.0, max_ang],
                'min_velocity': [-max_lin, 0.0, -max_ang],
            }
            ok, msg = self.send_params(smoother_params, client=self.smoother_params_client)
            if not ok:
                return False, f'Smoother update failed: {msg}'
        except Exception as e:
            return False, f'Failed to update smoother params: {e}'

        return True, 'OK'

    def send_params(self, params: dict, client=None):
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

    def make_param_value(self, value) -> ParameterValue:
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
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            pv.type = ParameterType.PARAMETER_STRING_ARRAY
            pv.string_array_value = value
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