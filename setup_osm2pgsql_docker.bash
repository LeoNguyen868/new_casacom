#!/bin/bash
set -euo pipefail

# Download Morocco OSM data if not already present
if [ ! -f ./morocco-latest.osm.pbf ]; then
    echo "Downloading Morocco OSM data..."
    wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf -O ./morocco-latest.osm.pbf
fi

# Setup PostgreSQL directories
echo "Setting up PostgreSQL directories..."
mkdir -p /var/run/postgresql
chown postgres:postgres /var/run/postgresql
chmod 2777 /var/run/postgresql

# Detect installed PostgreSQL major version
PG_BIN_DIR=$(dirname $(command -v psql) 2>/dev/null)
if [ -z "$PG_BIN_DIR" ]; then
  echo "psql not found in PATH"; exit 1
fi
PG_DIR=$(dirname "$PG_BIN_DIR")
PG_VERSION=$(ls -1 "$PG_DIR"/lib/postgresql | sed -n '1p' >/dev/null 2>&1; echo ${PG_VERSION_OVERRIDE:-})

# Fallback: parse version from binaries path
if [ -z "$PG_VERSION" ]; then
  # Try common versions
  for v in 17 16 15 14 13; do
    if [ -x "/usr/lib/postgresql/$v/bin/initdb" ]; then
      PG_VERSION=$v
      break
    fi
  done
fi

if [ -z "$PG_VERSION" ]; then
  echo "Unable to detect PostgreSQL version"; exit 1
fi
echo "Detected PostgreSQL version: $PG_VERSION"

# Initialize PostgreSQL database if not already done
if [ ! -d "/var/lib/postgresql/$PG_VERSION/main" ]; then
    echo "Initializing PostgreSQL database..."
    mkdir -p /var/lib/postgresql/$PG_VERSION
    chown postgres:postgres /var/lib/postgresql/$PG_VERSION
    su - postgres -c "/usr/lib/postgresql/$PG_VERSION/bin/initdb -D /var/lib/postgresql/$PG_VERSION/main"
    
    # Configure PostgreSQL for local trust authentication (no password required)
    echo "local all all trust" >> /var/lib/postgresql/$PG_VERSION/main/pg_hba.conf
    echo "host all all 127.0.0.1/32 trust" >> /var/lib/postgresql/$PG_VERSION/main/pg_hba.conf
fi

# Start PostgreSQL service and wait until ready
echo "Starting PostgreSQL service..."
su - postgres -c "/usr/lib/postgresql/$PG_VERSION/bin/pg_ctl -D /var/lib/postgresql/$PG_VERSION/main -w -t 60 start"

# Ensure log directory exists and configure minimal logging to avoid console spam
# Reset any previous log suppression so defaults apply
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET log_min_messages;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET client_min_messages;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET log_checkpoints;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET log_connections;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET log_disconnections;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET log_destination;\""
su - postgres -c "psql -v ON_ERROR_STOP=1 -d postgres -c \"ALTER SYSTEM RESET logging_collector;\""
su - postgres -c "psql -d postgres -c 'SELECT pg_reload_conf();'"

# No password setup needed when using 'trust' authentication

# Check if osm database exists and skip import if it does
echo "Checking for existing OSM database..."
if su - postgres -c "psql -lqt | cut -d \| -f 1 | grep -qw osm"; then
    echo "OSM database already exists. Skipping import."
    exit 0
fi

# Create OSM database with PostGIS extensions
echo "Creating osm database..."
su - postgres -c "createdb --template=template0 --encoding=UTF8 --owner=postgres osm"
su - postgres -c "psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS postgis;'"
su - postgres -c "psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS hstore;'"

# Import OSM data using osm2pgsql
echo "Importing OSM data with osm2pgsql..."
su - postgres -c "osm2pgsql -d osm -U postgres --create --slim --cache 4000 --number-processes 8 /app/morocco-latest.osm.pbf"

echo "complete"