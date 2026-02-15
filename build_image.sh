set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 [--version VERSION]"
    exit 1
}

# Default to a testing version for building stuff on the fly
VERSION="testing"

if [ $# -gt 1 ]; then
    usage
elif [ $# -eq 1 ]; then
    VERSION="$1"
fi

docker buildx build . -f docker/rbe_capstone.dockerfile --network=host --platform linux/amd64,linux/arm64 -t ghcr.io/sushiraj99/rbe-capstone:${VERSION}
