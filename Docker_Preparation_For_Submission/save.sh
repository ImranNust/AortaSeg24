#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set default container name
container_tag="name_of_your_docker_image"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    container_tag="$1"
fi

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$container_tag")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $container_tag"
    exit 1
fi

# Format the build information to remove special characters
# formatted_build_info=$(date -d "$build_timestamp" +"%Y%m%d_%H%M%S") # Not for mac
formatted_build_info=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$build_timestamp" +"%Y%m%d_%H%M%S") # For mac

# Set the output filename with timestamp and build information
output_filename="${SCRIPT_DIR}/${container_tag}_${formatted_build_info}.tar.gz"

# Save the Docker container and gzip it
docker save "$container_tag" | gzip -c > "$output_filename"

echo "Container saved as ${output_filename}"
