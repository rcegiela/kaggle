#!/bin/bash

# Usage: ./scp_upload.sh /path/to/local/file

# ===== CONFIGURATION =====
REMOTE_USER="azureuser"
REMOTE_HOST="20.171.146.103"
SSH_KEY="~/.ssh/vm-kaggle-gpu_key.pem"
REMOTE_DIR="~/kaggle"

# ===== CHECK ARGUMENT =====
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/local/file"
    exit 1
fi

LOCAL_FILE="$1"

# ===== PERFORM SCP =====
echo "[INFO] Uploading $LOCAL_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
scp -i "$SSH_KEY" "$LOCAL_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# ===== DONE =====
if [ $? -eq 0 ]; then
    echo "[INFO] File transfer complete."
else
    echo "[ERROR] File transfer failed."
fi