#!/bin/bash

# Usage:
# ./deploy_to_vm.sh <SSH_KEY_PATH> <USERNAME> <HOSTNAME> [REMOTE_DIR]

SSH_KEY=$1
USER=$2
HOST=$3
REMOTE_DIR=${4:-/home/$USER}
IMAGE_NAME="ml-env"
IMAGE_TAR="ml-env.tar"
REMOTE_SCRIPT="run_ml_container.sh"

# --- Validate input ---
if [[ -z $SSH_KEY || -z $USER || -z $HOST ]]; then
    echo "Usage: ./deploy_to_vm.sh <ssh_key_path> <username> <hostname> [remote_dir]"
    exit 1
fi

# --- Step 1: Build/Update the Docker image ---
echo "📦 Building/Updating Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

if [[ $? -ne 0 ]]; then
    echo "❌ Docker build failed. Aborting."
    exit 1
fi

# --- Step 2: Save Docker image to tar ---
echo "📦 Exporting image to $IMAGE_TAR"
docker save "$IMAGE_NAME" -o "$IMAGE_TAR"

# --- Step 3: Copy tar and script to remote VM ---
echo "📤 Copying image and script to $USER@$HOST:$REMOTE_DIR ..."
scp -i "$SSH_KEY" "$IMAGE_TAR" "$REMOTE_SCRIPT" "$USER@$HOST:$REMOTE_DIR/"

# --- Step 4: SSH and run the remote script ---
echo "🔐 Connecting via SSH and running container setup..."
ssh -i "$SSH_KEY" "$USER@$HOST" "cd $REMOTE_DIR && chmod +x $REMOTE_SCRIPT && ./$REMOTE_SCRIPT"
