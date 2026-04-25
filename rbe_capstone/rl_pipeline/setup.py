from setuptools import setup
from glob import glob
import os

package_name = 'rl_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    description='RL pipeline package',
    license='TODO',
    entry_points={
        'console_scripts': [
            'planner_config_manager = rl_pipeline.planner_config_manager:main',
            'rl_backbone = rl_pipeline.rl_backbone:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/rl_pipeline']),
        ('share/rl_pipeline', ['package.xml']),
        (os.path.join('share', 'rl_pipeline', 'launch'),
            glob('launch/*.launch.py')),
    ],
)