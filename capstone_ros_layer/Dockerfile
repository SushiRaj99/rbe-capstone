FROM ros:jazzy-ros-core

# ROS2 setup
COPY ./cyclonedds.xml root/.ros/cyclonedds.xml
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=.ros/cyclonedds.xml

ENV TERM=xterm-color
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
 && echo "[ -f /root/ws/install/setup.bash ] && source /root/ws/install/setup.bash" >> /root/.bashrc \
 && echo "export RCUTILS_COLORIZED_OUTPUT=1" >> /root/.bashrc

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
  ros-jazzy-ament-cmake \
  ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-turtlebot4-description \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nav2-simple-commander \
  ros-jazzy-rviz2 \
  ros-jazzy-joint-state-publisher-gui \
  nano \
  vim-gtk3 \
  cscope \ 
  universal-ctags \
  && apt clean

# Additional configuration for development
COPY .vimrc /root/.vimrc
COPY build_cscope_ws.sh /root/build_cscope_ws.sh
RUN chmod +x /root/build_cscope_ws.sh
RUN echo "alias build_cscope_ws='/root/build_cscope_ws.sh'" >> /root/.bashrc

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && colcon mixin update"
