set -euo pipefail

usage() {
    echo "Usage: $0 [version]"
    exit 1
}

# Default to a testing version for building stuff on the fly
VERSION="testing"
loadcmd=()

# Process optional arguments (e.g. take in version as an optional argument)
while [ $# -ge 1 ]; do
    case $1 in
        '-v'|'-V'|'--version')
            shift
            VERSION="${1}"
            ;;
        '-l'|'-L'|'--load')
            loadcmd=(--load)
            ;;
        '-h'|'--help')
            echo "build_ros_layer.sh contains the following optional inputs:"
            echo -e "\t-v|-V|--version\n\t\t[Arguments: 1] Specifies the version tag for the built docker image (default tag is 'testing')." | fmt
            echo -e "\t-l|-L|--load\n\t\t[Arguments: 0] Specifies flag to load the built docker image into the local docker image registry." | fmt
            exit
            ;;
        *)
            echo "ERROR: Invalid input argument. Seek help via 'build_ros_layer.sh --help'"
            exit 1
            ;;
    esac
    shift;
done

docker buildx build . --network=host --platform linux/amd64,linux/arm64 -t ghcr.io/sushiraj99/rbe-capstone/capstone_ros_layer:${VERSION} ${loadcmd[@]}