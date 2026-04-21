FROM ghcr.io/sushiraj99/rbe-capstone/capstone_ros_layer:0.0.2

# Install Python RL dependencies
RUN apt update && apt install -y python3-pip && apt clean
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --break-system-packages

# Copy source packages into workspace
COPY rbe_capstone/ /root/ws/src/rbe_capstone/

# Build the workspace, then strip source/build artifacts — only install/ is needed at runtime
RUN /bin/bash -c " \
  source /opt/ros/jazzy/setup.bash && \
  cd /root/ws && \
  colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && \
  rm -rf src build log"
