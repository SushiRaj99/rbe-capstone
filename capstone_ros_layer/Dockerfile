FROM ros:jazzy-ros-core

# ROS2 setup
COPY ./cyclonedds.xml root/.ros/cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/root/.ros/cyclonedds.xml

ENV TERM=xterm-color
RUN echo "export RCUTILS_COLORIZED_OUTPUT=1" >> /root/.bashrc

# Add entrypoint script
COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh

RUN apt update && apt upgrade -y \
  && apt clean

# Install ROS2 build tools and dependencies
RUN apt update && apt install -y \
  build-essential \
  git \
  python3-colcon-cmake \
  python3-colcon-common-extensions \
  python3-colcon-mixin \
  python3-colcon-ros \
  python3-rosdep \
  python3-vcstool \
  ros-$ROS_DISTRO-ament-cmake \
  ros-$ROS_DISTRO-rmw-cyclonedds-cpp \
  ros-$ROS_DISTRO-rviz2 \
  nano \
  vim-gtk3 \
  cscope \ 
  universal-ctags \
  curl \
  && apt clean
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Additional configuration for development
COPY .vimrc /root/.vimrc
COPY build_cscope_ws.sh /root/build_cscope_ws.sh
RUN chmod +x /root/build_cscope_ws.sh
RUN echo "alias build_cscope_ws='/root/build_cscope_ws.sh'" >> /root/.bashrc
RUN echo "alias init_jazzy='source /opt/ros/${ROS_DISTRO}/setup.bash'" >> /root/.bashrc
RUN echo "colconbuild()\n{\n\tpushd /root/ws >/dev/null && colcon build \"\$@\" && popd > /dev/null\n}" >> /root/.bashrc
RUN echo "code()\n{\n\tcode-server \"\$@\" --auth=none\n}" >> /root/.bashrc

# Setup the entrypoint and shared workspace
RUN mkdir -p /root/ws/src/rbe_capstone
WORKDIR /root/ws
RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && colcon mixin update"
