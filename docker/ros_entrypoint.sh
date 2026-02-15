#!/bin/bash
set -e
# Go to Home by default
cd root
# setup ros environment
#source "/root/ws/install/setup.bash"
exec "$@"