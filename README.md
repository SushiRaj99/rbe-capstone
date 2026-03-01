# RBE-Capstone

## Docker

A lot of this project will revolve around using docker. Docker is a good fit for us given the different platforms used by the team (Mac/Windows/Linux etc). It helps us set up the dependencies and environment once and make sure we have an apples to apples comparison when working on developing together.

This [tutorial](https://docker-curriculum.com/#docker-compose) is good for explaining why Docker is useful and how to use it.

We are utilizing two layers for building our docker images. The first is the **capstone_ros_layer** - This layer contains the dependencies we install via apt and other sources. Isolating the layer where we install ROS, ROS packages, and other dependencies helps us avoid wasting time when building new images with code changes.

To update the capstone_ros_layer, edit `capstone_ros_layer/Dockerfile`, and build a new image by running `./build_ros_layer.sh`.

**Note** - Windows users will need to run the following command in WSL before using the build scripts (in order to set up multi-platform builds).

```
docker buildx create --use --name multiarch --driver docker-container
docker run --privileged --rm tonistiigi/binfmt --install all

```

Our code will live in the `rbe_capstone` folder, which gets built and baked into the **rbe_capstone** image. This image gets built on top of the **capstone_ros_layer**. The rbe_capstone image can be built by running the `build_image.sh` script. This image pulls in the code from the rbe_capstone folder, builds it, and then deletes the source code, build logs, and build files from the image. The necessary binaries are preserved in order to make the image as lightweight as possible.

An entrypoint script script and some additions to the Dockerfiles are made in order to source the relevant workspaces, this allows us to use ROS tools and our code as soon as we enter the container.

Separate `docker-compose.yml` files are set up for Windows and Mac, this is needed because different setups are needed for being able to use different visualization tools (like rviz2). The docker compose files mount the working directory into  `/root/ws/src/`, this means that whatever changes you make while working will make their way into the container, making development easier.

## Shut up and tell me how to develop

As mentioned above, our code will live in the rbe_capstone directory. Our code will mostly be contained in ROS2 packages ([tutorial for setting up ROS2 packages](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)). You can can modify your code in your usual editor (like VSCode) and build within the container.

You can build all the packages in the workspace with:

```
cd /root/ws/
colcon build
# When we get to the point where we have multiple packages, you can build a single package to save time with
colcon build --packages-select <package_name>
```

After building, you can make sure that that your changes are in effect by running

```
cd /root/ws
source install/setup.bash
```

## Development Workflow

### Bringing up the container

You can bring up the docker container with the following command

```
# Mac
docker compose up -d
# Windows
docker compose -f wsl_entrypoint/docker-compose-wsl.yml up -d
```

### Entering the container

```
docker exec -it rbe_capstone bash
```

From there, you can develop your code and build as needed. Whenever you feel that you have made changes that you think are ready to commit to, you can release a new image and push. You can build a new image with

```
./build_image.sh <optional image tag>
```

The optional image tag will default to "testing", this means that the new image will be named `ghcr.io/sushiraj99/rbe-capstone:image_tag`

The image used when bringing up the container is set in the docker compose file:

```
services:
  rbe_capstone:
    image: ghcr.io/sushiraj99/rbe-capstone:my_image_tag
    ...
```

When you have a new image that we want everyone to start using, you will need to update this line in the docker-compose files. You can push a new image up to the cloud by running:

```
docker push ghcr.io/sushiraj99/rbe-capstone:my_image_tag
```

You can see what's in the cloud on GitHub[ here](https://github.com/SushiRaj99?tab=packages&repo_name=rbe-capstone). You can pull down new images with the following command:

```
docker pull ghcr.io/sushiraj99/rbe-capstone:my_image_tag
```

If you haven't downloaded an image, starting the container will also pull down the image.

## ROS2 Packages

### Simulation Launch

Basically, what the name says. This package is basically just for launching visualization and simulation. Right now, this holds the setup to visualize the robot, a map and launch RVIZ2 for simulation. This can be launched with:

```
ros2 launch simulation_launch view_robot.launch.py map_name:="warehouse"
```

The map_name argument is optional and will default to "warehouse". The maps must be stored in the `simulation_laumch/maps` folder with the format `<map_name>/<map_name>.pgm` and `<map_name>/<map_name>.yaml`. The defaults for the RVIZ2 visualizations are saved in the `rviz folder.
