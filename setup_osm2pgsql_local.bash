#!/bin/bash
set -euo pipefail

# Setup script for local OSM2pgsql environment (not in Docker)
# This script sets up PostgreSQL with PostGIS and imports Morocco OSM data

echo "=== Local OSM2pgsql Setup Script ==="

# Check if required tools are installed
echo "Checking required tools..."

# Check PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL client not found. Please install PostgreSQL."
    echo "On Ubuntu/Debian: sudo apt-get install postgresql postgresql-contrib"
    exit 1
fi

# Check osm2pgsql
if ! command -v osm2pgsql &> /dev/null; then
    echo "osm2pgsql not found. Please install osm2pgsql."
    echo "On Ubuntu/Debian: sudo apt-get install osm2pgsql"
    exit 1
fi

# Check wget for downloading data
if ! command -v wget &> /dev/null; then
    echo "wget not found. Please install wget."
    echo "On Ubuntu/Debian: sudo apt-get install wget"
    exit 1
fi

echo "All required tools are installed."

# Download Morocco OSM data if not already present
if [ ! -f "./morocco-latest.osm.pbf" ]; then
    echo "Downloading Morocco OSM data..."
    wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf -O ./morocco-latest.osm.pbf
else
    echo "Morocco OSM data already exists, skipping download."
fi

# Detect installed PostgreSQL major version
echo "Detecting PostgreSQL version..."
PG_VERSION=$(psql --version | awk '{print $3}' | cut -d. -f1)
echo "Detected PostgreSQL version: $PG_VERSION"

# Check if PostgreSQL service is running
echo "Checking PostgreSQL service status..."
if ! pg_isready -h localhost > /dev/null 2>&1; then
    echo "PostgreSQL service is not running."
    echo "Please start PostgreSQL service:"
    echo "  sudo systemctl start postgresql"
    echo "  sudo systemctl enable postgresql"
    exit 1
fi
echo "PostgreSQL service is running."

# Check if osm database exists and skip import if it does
echo "Checking for existing OSM database..."
if sudo -u postgres psql -lqt | cut -d \| -f 1 | grep -qw osm; then
    echo "OSM database already exists. Skipping setup."
    echo "To recreate the database, run: sudo -u postgres dropdb osm"
    exit 0
fi

# Create OSM database with PostGIS extensions
echo "Creating osm database..."
sudo -u postgres createdb osm
sudo -u postgres psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS postgis;'
sudo -u postgres psql -d osm -c 'CREATE EXTENSION IF NOT EXISTS hstore;'

# Import OSM data using osm2pgsql
echo "Importing OSM data with osm2pgsql..."
echo "This may take a while depending on your system performance..."

sudo -u postgres osm2pgsql -d osm -U postgres --create --slim --cache 4000 --number-processes $(nproc) ./morocco-latest.osm.pbf

echo "=== Setup Complete ==="
echo "OSM database 'osm' has been created and populated with Morocco data."
echo "You can now connect to it using: psql -d osm"
