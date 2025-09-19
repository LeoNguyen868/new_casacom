#!/bin/bash

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

# Initialize PostgreSQL database if not already done
if [ ! -d "/var/lib/postgresql/17/main" ]; then
    echo "Initializing PostgreSQL database..."
    mkdir -p /var/lib/postgresql/17
    chown postgres:postgres /var/lib/postgresql/17
    su - postgres -c "/usr/lib/postgresql/17/bin/initdb -D /var/lib/postgresql/17/main"
    
    # Configure PostgreSQL for password authentication
    echo "host all all 127.0.0.1/32 md5" >> /var/lib/postgresql/17/main/pg_hba.conf
    echo "local all all md5" >> /var/lib/postgresql/17/main/pg_hba.conf
fi

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
service postgresql start
sleep 1  # brief pause

# Wait until PostgreSQL is ready to accept connections
echo "Waiting for PostgreSQL to become ready..."
for i in {1..60}; do
    if pg_isready -h localhost -p 5432 -U postgres > /dev/null 2>&1; then
        echo "PostgreSQL is ready."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "PostgreSQL did not become ready in time. Showing status and exiting." >&2
        su - postgres -c "/usr/lib/postgresql/17/bin/pg_ctl -D /var/lib/postgresql/17/main status || true"
        exit 1
    fi
    sleep 1
done

# Set postgres user password
echo "Setting postgres user password..."
su - postgres -c "psql -c \"ALTER USER postgres WITH PASSWORD 'postgres';\""

# Check if osm database exists and drop it if it does
echo "Checking for existing OSM database..."
if su - postgres -c "psql -lqt | cut -d \| -f 1 | grep -qw osm"; then
    echo "Dropping existing osm database..."
    su - postgres -c "dropdb osm"
fi

# Create OSM database with PostGIS extensions
echo "Creating osm database..."
su - postgres -c "createdb --encoding=UTF8 --owner=postgres osm"
su - postgres -c "psql -d osm -c 'CREATE EXTENSION postgis;'"
su - postgres -c "psql -d osm -c 'CREATE EXTENSION hstore;'"

# Import OSM data using osm2pgsql
echo "Importing OSM data with osm2pgsql..."
# Use TCP connection and password auth to avoid unix socket issues
su - postgres -c "PGPASSWORD=postgres osm2pgsql -H localhost -P 5432 -d osm -U postgres --create --slim --cache 4000 --number-processes 8 /app/morocco-latest.osm.pbf"

echo "OSM data import completed successfully!"