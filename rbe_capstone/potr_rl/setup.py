from setuptools import setup, find_packages

setup(
    name='potr_rl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'stable-baselines3',
        'numpy',
    ],
)
