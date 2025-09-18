#!/bin/bash

# Run the specified command (default is download_maroc.py)
echo "Starting command: $@"
"$@"

echo ""
echo "==============================================="
echo "Download process completed."
echo "Container is still running to preserve data."
echo "Use Ctrl+C or 'docker-compose down' to stop it."
echo "==============================================="
echo ""

# Keep container running
tail -f /dev/null
