#!/bin/bash

# Make sure environment variables are properly set
echo "Environment setup:"
echo "DATA_RAW_PATH: $DATA_RAW_PATH"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "RESULT_DIR: $RESULT_DIR"

# Set the dates to process
if [ $# -eq 0 ]; then
    DATES=("2025-08-05" "2025-08-06" "2025-08-07" "2025-08-9" "2025-08-10" "2025-08-11" "2025-08-12" "2025-08-13" "2025-08-14" "2025-08-15" "2025-08-16" "2025-08-17" "2025-08-18" "2025-08-19" "2025-08-20" "2025-08-21" "2025-08-22" "2025-08-23" "2025-08-24" "2025-08-25" "2025-08-26" "2025-08-27" "2025-08-28" "2025-08-29" "2025-08-30" "2025-08-31" "2025-09-01" "2025-09-02" "2025-09-03" "2025-09-04" "2025-09-05" "2025-09-06" "2025-09-07" "2025-09-08" "2025-09-09" "2025-09-10" "2025-09-11" "2025-09-12" "2025-09-13" "2025-09-14" "2025-09-15" "2025-09-16" "2025-09-17")
else
    DATES=("$@")
fi

# Create PGPASS file for passwordless authentication
echo "Setting up PostgreSQL authentication..."
echo "localhost:5432:*:postgres:postgres" > ~/.pgpass
chmod 600 ~/.pgpass
export PGPASSWORD=postgres

# Always ensure PostgreSQL is initialized and running, and OSM DB is prepared
echo "Ensuring PostgreSQL and OSM database are ready..."
/app/setup_osm2pgsql_docker.bash

# Process each date in the list
for DATE in "${DATES[@]}"; do
    # Validate date format (YYYY-MM-DD)
    if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid date format: $DATE. Please use YYYY-MM-DD format."
        continue
    fi

    # Run download_maroc.py for a specific date
    echo "Starting data download for $DATE..."
    python download_maroc.py --start-date "$DATE" --end-date "$DATE"

    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Download completed successfully for $DATE."
    else
        echo "Download failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Starting data processing for $DATE..."
    python process_maroc.py --start-date "$DATE" --end-date "$DATE"

    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Processing completed successfully for $DATE."
    else
        echo "Processing failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Removing raw data for $DATE..."
    # Use quotes around the path to handle potential spaces in the path
    rm -rf "${DATA_RAW_PATH}/$DATE"

    # Check if raw data was removed
    if [ $? -eq 0 ]; then
        echo "Raw data removed successfully for $DATE."
    else
        echo "Failed to remove raw data for $DATE. Check folder permissions."
    fi
    
    echo "Completed pipeline for $DATE"
    echo "----------------------------------------"
done

echo "Processing POI"
python process_maid.py

echo "Completed pipeline for POI"
echo "----------------------------------------"

echo "Merging results"
python merge.py

echo "Completed pipeline for merging results"
echo "----------------------------------------"

# Keep the container running to maintain the PostgreSQL server and allow users to access results
echo "Pipeline processing completed. Container will remain running."
echo "Use Ctrl+C or 'docker compose down' to stop the container."

# Keep container running
tail -f /dev/null