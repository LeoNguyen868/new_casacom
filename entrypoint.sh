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

# Ensure PostgreSQL service is running (idempotent)
echo "Ensuring PostgreSQL service is running..."
mkdir -p /var/run/postgresql
chown postgres:postgres /var/run/postgresql
chmod 2777 /var/run/postgresql

# Detect installed PostgreSQL version directory
PG_VERSION_DETECTED=""
for v in 17 16 15 14 13 12 11; do
  if [ -x "/usr/lib/postgresql/$v/bin/pg_ctl" ] && [ -d "/var/lib/postgresql/$v/main" ]; then
    PG_VERSION_DETECTED=$v
    break
  fi
done

if [ -n "$PG_VERSION_DETECTED" ]; then
  if ! su - postgres -c "/usr/lib/postgresql/$PG_VERSION_DETECTED/bin/pg_ctl -D /var/lib/postgresql/$PG_VERSION_DETECTED/main status" >/dev/null 2>&1; then
    echo "Starting PostgreSQL (version $PG_VERSION_DETECTED)..."
    su - postgres -c "/usr/lib/postgresql/$PG_VERSION_DETECTED/bin/pg_ctl -D /var/lib/postgresql/$PG_VERSION_DETECTED/main -w start"
  else
    echo "PostgreSQL already running."
  fi

  # Ensure 'osm' database exists; if not, create and import
  if ! su - postgres -c "psql -lqt | cut -d '|' -f 1 | grep -qw osm"; then
    echo "'osm' database not found. Creating and importing OSM data..."
    su - postgres -c "createdb --template=template0 --encoding=UTF8 --owner=postgres osm"
    su - postgres -c "psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS postgis;'"
    su - postgres -c "psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS hstore;'"
    if [ -f "/app/morocco-latest.osm.pbf" ]; then
      su - postgres -c "osm2pgsql -d osm -U postgres --create --slim --cache 4000 --number-processes 8 /app/morocco-latest.osm.pbf"
    else
      echo "OSM PBF file not found at /app/morocco-latest.osm.pbf; skipping import."
    fi
  fi
else
  echo "Warning: Could not detect PostgreSQL installation directory; skipping start attempt."
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
