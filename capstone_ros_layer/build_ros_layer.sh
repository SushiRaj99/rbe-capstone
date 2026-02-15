set -euo pipefail

usage() {
    echo "Usage: $0 [version]"
    exit 1
}

# Default to a testing version for building stuff on the fly
VERSION="testing"

# Take in version as an optional argumen
if [ $# -gt 1 ]; then
    usage
elif [ $# -eq 1 ]; then
    VERSION="$1"
fi

docker buildx build . --network=host --platform linux/amd64,linux/arm64 -t ghcr.io/sushiraj99/rbe-capstone/capstone_ros_layer:${VERSION}
