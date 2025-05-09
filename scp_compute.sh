#!/bin/bash

# Usage: ./scp_upload.sh /path/to/local/file_or_dir_or_pattern

# ===== CONFIGURATION =====
REMOTE_USER="azureuser"
REMOTE_HOST="57.154.208.193"
SSH_KEY="$HOME/.ssh/vm-compute_key.pem"
REMOTE_DIR="~/kaggle"

# ===== CHECK ARGUMENT =====
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/local/file_or_dir_or_pattern"
    exit 1
fi

LOCAL_PATH="$1"

# ===== PERFORM RSYNC =====
echo "[INFO] Uploading $LOCAL_PATH to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
rsync -avz -e "ssh -i $SSH_KEY" $LOCAL_PATH "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"

# ===== DONE =====
if [ $? -eq 0 ]; then
    echo "[INFO] Transfer complete."
else
    echo "[ERROR] Transfer failed."
fi