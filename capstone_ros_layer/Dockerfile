FROM ros:jazzy-ros-core

# ROS2 setup
COPY ./cyclonedds.xml root/.ros/cyclonedds.xml

ENV TERM=xterm-color
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
 && echo "[ -f /root/ws/install/setup.bash ] && source /root/ws/install/setup.bash" >> /root/.bashrc \
 && echo "export RCUTILS_COLORIZED_OUTPUT=1" >> /root/.bashrc

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
  python3-pip \
  python3-venv \
  ros-jazzy-ament-cmake \
  ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-turtlebot4-description \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nav2-simple-commander \
  ros-jazzy-rviz2 \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-rqt-graph \
  nano \
  vim-gtk3 \
  cscope \
  universal-ctags \
  && apt clean
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN code-server --install-extension Ranch-Hand-Robotics.rde-ros-2
RUN python3 -m venv /opt/.venv --system-site-packages && \
  /opt/.venv/bin/pip install --no-cache-dir --upgrade pip && \
  /opt/.venv/bin/pip install --no-cache-dir "stable-baselines3[extra]" && \
  /opt/.venv/bin/pip install --no-cache-dir gymnasium
RUN touch /opt/.venv/COLCON_IGNORE

# Additional configuration for development
COPY .vimrc /root/.vimrc
COPY build_cscope_ws.sh /root/build_cscope_ws.sh
RUN chmod +x /root/build_cscope_ws.sh
RUN echo "alias build_cscope_ws='/root/build_cscope_ws.sh'" >> /root/.bashrc
RUN echo "alias init_jazzy='source /opt/ros/${ROS_DISTRO}/setup.bash'" >> /root/.bashrc
RUN echo "alias init_ws='[ -f /root/ws/install/setup.bash ] && source /root/ws/install/setup.bash'" >> /root/.bashrc
RUN echo "alias init_env='source /opt/.venv/bin/activate && init_jazzy && init_ws'" >> /root/.bashrc
RUN echo "colconbuild()\n{\n\tpushd /root/ws >/dev/null && colcon build \"\$@\" && popd > /dev/null\n}" >> /root/.bashrc
RUN echo "alias colconclean='pushd /root/ws >/dev/null && rm -rf build install log && popd > /dev/null'" >> /root/.bashrc
RUN echo "code()\n{\n\tcode-server \"\$@\" --auth=none --ignore-last-opened\n}" >> /root/.bashrc
RUN echo "export PATH=\"/opt/.venv/bin:\${PATH}\"" >> /root/.bashrc

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && colcon mixin update"
