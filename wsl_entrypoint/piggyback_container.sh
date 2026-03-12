#!/bin/bash

dockerps_str=$(docker ps)
if [ $# -gt 0 ]; then
	[ $# -gt 1 ] && echo "Warning: multiple arguments supplied to piggyback_container() - only the first argument will be used (\"${1}\")"
	dockerps_str=$(echo "${dockerps_str}" | grep "$1")
fi
dockerps_array=(${dockerps_str})
dpslen=${#dockerps_array[@]}
dpsend=${dockerps_array[((${dpslen}-1))]}

echo "Piggybacking off of docker container: ${dpsend}..."

docker exec -it ${dpsend} bash
