FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    groff \
    less \
    wget \
    git \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Add PostgreSQL repository
RUN curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list

# Install PostgreSQL, osm2pgsql and other dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    postgresql \
    postgresql-contrib \
    postgis \
    osm2pgsql \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install awscli-plugin-endpoint plotly shapely pyproj reverse_geocoder

# Copy the project files
COPY *.py /app/
COPY *.json /app/
COPY *.parquet /app/
COPY *.bash /app/
COPY *.pbf /app/
COPY *.pkl /app/
COPY *.feather /app/
# Create directories
RUN mkdir -p /app/data/raw /app/data/raw_rt /app/data/processed /app/result

# Setup AWS configuration
RUN mkdir -p /root/.aws
COPY aws_config.txt /root/.aws/config

# Set environment variables
ENV PATH="/app:${PATH}"
ENV DATA_RAW_PATH1="/app/data/raw"
ENV DATA_RAW_PATH2="/app/data/raw_rt"
ENV OUTPUT_DIR="/app/data/processed_all"
ENV POSTGRES_DB="osm"
ENV POSTGRES_USER="postgres"
ENV POSTGRES_PASSWORD="postgres"
ENV POSTGRES_HOST="127.0.0.1"
ENV POSTGRES_PORT="5432"


# Make scripts executable
RUN chmod +x /app/*.bash
