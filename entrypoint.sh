#!/bin/bash

# Initialize and start PostgreSQL + import OSM if not already done
if [ -z "$SKIP_OSM_SETUP" ]; then
    if [ ! -f /var/lib/postgresql/.osm_import_done ]; then
        echo "Running OSM/PostgreSQL setup..."
        /app/setup_osm2pgsql_docker.bash || { echo "OSM setup failed"; exit 1; }
        touch /var/lib/postgresql/.osm_import_done
    else
        echo "OSM/PostgreSQL already initialized. Skipping setup."
    fi
else
    echo "SKIP_OSM_SETUP is set; skipping OSM/PostgreSQL setup."
fi

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
