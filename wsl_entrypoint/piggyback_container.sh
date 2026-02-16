#!/bin/bash

dockerps_str=$(docker ps)
dockerps_array=(${dockerps_str})
dpslen=${#dockerps_array[@]}
dpsend=${dockerps_array[((${dpslen}-1))]}

echo "Piggybacking off of docker container: ${dpsend}..."

docker exec -it ${dpsend} bash
