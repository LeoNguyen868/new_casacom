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

# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git lfs install
