#!/bin/bash
set -euo pipefail

# Script to install required dependencies for OSM2pgsql setup

echo "=== Installing Dependencies for OSM2pgsql ==="

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install PostgreSQL and PostGIS
echo "Installing PostgreSQL and PostGIS..."
sudo apt-get install -y postgresql postgresql-contrib postgis

# Install osm2pgsql
echo "Installing osm2pgsql..."
sudo apt-get install -y osm2pgsql

# Install wget (usually already installed)
echo "Installing wget..."
sudo apt-get install -y wget

# Start PostgreSQL service
echo "Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql

echo "=== Installation Complete ==="
echo "Dependencies have been installed successfully."
echo "You can now run: ./setup_osm2pgsql_local.bash"
