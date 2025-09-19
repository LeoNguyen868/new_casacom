#!/bin/bash

# Create necessary directories matching Dockerfile.pipeline and docker-compose.yml
mkdir -p ./data/raw_rt
mkdir -p ./data/processed
mkdir -p ./result

# Setup AWS CLI configuration
mkdir -p ~/.aws
cat > ~/.aws/config << EOL
[profile hw]
region = gra
endpoint_url = https://s3.gra.io.cloud.ovh.net
EOL

# Create AWS credentials file
mkdir -p ~/.aws
touch ~/.aws/credentials

# Set environment variables matching Dockerfile.pipeline and docker-compose.yml
export AWS_PROFILE=hw
export DATA_RAW_PATH=/app/data/raw_rt
export DATA_RAW_PATH1=/app/data/raw_rt
export DATA_RAW_PATH2=/app/data/raw
export OUTPUT_DIR=/app/data/processed
export RESULT_DIR=/app/result
export POSTGRES_DB=osm
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432

# Optional: Add AWS CLI endpoint plugin
aws configure set plugins.endpoint awscli-plugin-endpoint

echo "Local environment setup completed successfully."
echo "Directories and environment variables configured to match Docker setup."
