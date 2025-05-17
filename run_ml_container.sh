#!/bin/bash

IMAGE_TAR="ml-env.tar"
IMAGE_NAME="ml-env"
CONTAINER_NAME="ml-dev"

# Step 1: Load the image
echo "🔄 Loading image from $IMAGE_TAR..."
docker load -i "$IMAGE_TAR"

# Step 2: Check for existing container, or existing running container
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "✅ Container $CONTAINER_NAME is already running. Attaching..."
    docker attach "$CONTAINER_NAME"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🟡 Container $CONTAINER_NAME exists but stopped. Starting..."
    docker start -ai "$CONTAINER_NAME"
else
    echo "🚀 Creating container $CONTAINER_NAME with GPU access..."
    docker run -it \
        --name "$CONTAINER_NAME" \
        --gpus all \
        -v "$PWD:/ml" \
        -w /ml \
        "$IMAGE_NAME" \
        bash
fi
