#!/bin/bash

# If no command is provided, run the pipeline by default
if [ $# -eq 0 ]; then
    echo "No command provided. Running default pipeline: /app/docker_pipeline.bash"
    /app/docker_pipeline.bash
else
    echo "Starting command: $@"
    "$@"
fi

# Keep container running after the task completes (to inspect results/logs)
echo ""
echo "==============================================="
echo "Task completed. Container will remain running."
echo "Use Ctrl+C or 'docker compose down' to stop it."
echo "==============================================="
echo ""

tail -f /dev/null
