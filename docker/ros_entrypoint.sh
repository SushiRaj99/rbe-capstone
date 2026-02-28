#!/bin/bash
set -e
# Go to Home by default
cd root
# setup ros environment
source /opt/ros/jazzy/setup.bash
[ -f "/root/ws/install/setup.bash" ] && source "/root/ws/install/setup.bash"
exec "$@"