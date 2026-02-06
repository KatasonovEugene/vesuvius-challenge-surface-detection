#!/bin/bash
set -e

ARCHIVE_NAME="../vesuvius-challenge-surface-detection.tar.gz"
SRC_DIR="."

tar \
  --exclude="./ext" \
  --exclude="./.git" \
  --exclude="./data" \
  --exclude="./wandb" \
  --exclude="./outputs" \
  --exclude="./saved" \
  --exclude="./*.tar.gz" \
  -czvf "$ARCHIVE_NAME" \
  "$SRC_DIR"
