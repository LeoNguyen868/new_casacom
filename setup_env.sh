#!/bin/bash

# Install system dependencies
sudo apt-get update && sudo apt-get install -y \
    curl \
    unzip \
    wget \
    git \
    postgresql-client \
    postgresql \
    postgis \
    python3-pip

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws

# Install Python dependencies
pip3 install awscli-plugin-endpoint plotly shapely pyproj reverse_geocoder

# Create necessary directories 
mkdir -p ~/data/raw_rt
mkdir -p ~/data/processed
mkdir -p ~/result

# Setup AWS CLI configuration
mkdir -p ~/.aws
cat > ~/.aws/config << EOL
[profile hw]
region = gra
endpoint_url = https://s3.gra.io.cloud.ovh.net
EOL

# Create AWS credentials file
touch ~/.aws/credentials

# Set environment variables 
export AWS_PROFILE=hw
export DATA_RAW_PATH1="~/data/raw_rt"
export DATA_RAW_PATH2="~/data/raw"
export OUTPUT_DIR="~/data/processed"
export RESULT_DIR="~/result"
export POSTGRES_DB="osm"
export POSTGRES_USER="postgres"
export POSTGRES_PASSWORD="postgres"
export POSTGRES_HOST="localhost"
export POSTGRES_PORT="5432"

# Add AWS CLI endpoint plugin
aws configure set plugins.endpoint awscli-plugin-endpoint

echo "Local environment setup completed successfully."
echo "Dependencies and environment variables configured."
