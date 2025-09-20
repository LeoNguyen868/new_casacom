#!/bin/bash

# Make sure environment variables are properly set
echo "Environment setup:"
echo "DATA_RAW_PATH1: $DATA_RAW_PATH1"
echo "DATA_RAW_PATH2: $DATA_RAW_PATH2"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "RESULT_DIR: $RESULT_DIR"

# Set the dates to process
if [ $# -eq 0 ]; then
    DATES1=(
     "2025-07-31"
    )
    DATES2=(
        "2025-08-22"
        "2025-08-23"
        "2025-08-24"
        "2025-08-25"
        "2025-08-26"
        "2025-08-27"
        "2025-08-28"
        "2025-08-29"
        "2025-08-30"
        "2025-08-31"
        "2025-09-01"
        "2025-09-02"
        "2025-09-03"
        "2025-09-04"
        "2025-09-05"
        "2025-09-06"
        "2025-09-07"
        "2025-09-08"
        "2025-09-09"
        "2025-09-10"
        "2025-09-11"
        "2025-09-12"
        "2025-09-13"
        "2025-09-14"
        "2025-09-15"
        "2025-09-16"
        "2025-09-17"
        "2025-09-18"
    )
else
    # Use provided arguments for both DATES1 and DATES2
    DATES1=("$@")
    DATES2=("$@")
fi

# Process DATES1 first
for DATE in "${DATES1[@]}"; do
    # Validate date format (YYYY-MM-DD)
    if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid date format: $DATE. Please use YYYY-MM-DD format."
        continue
    fi

    # Run download_maroc.py for a specific date
    echo "Starting data download for $DATE..."
    python3 download_maroc.py --start-date "$DATE" --end-date "$DATE" --local-dir ./data/raw/ --bucket s3://home-work/hw/project2

    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Download completed successfully for $DATE."
    else
        echo "Download failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Starting data processing for $DATE..."
    python3 process_maroc.py --start-date "$DATE" --end-date "$DATE" --raw-data ./data/raw/

    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Processing completed successfully for $DATE."
    else
        echo "Processing failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Removing raw data for $DATE..."
    # Use quotes around the path to handle potential spaces in the path
    rm -rf "${DATA_RAW_PATH1}/$DATE"

    # Check if raw data was removed
    if [ $? -eq 0 ]; then
        echo "Raw data removed successfully for $DATE."
    else
        echo "Failed to remove raw data for $DATE. Check folder permissions."
    fi
    
    echo "Completed pipeline for $DATE"
    echo "----------------------------------------"
done

# Then process DATES2
for DATE in "${DATES2[@]}"; do
    # Validate date format (YYYY-MM-DD)
    if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid date format: $DATE. Please use YYYY-MM-DD format."
        continue
    fi

    # Run download_maroc.py for a specific date
    echo "Starting data download for $DATE..."
    python3 download_maroc.py --start-date "$DATE" --end-date "$DATE" --local-dir ./data/raw_rt/ --bucket s3://home-work/hw/project2_rt

    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Download completed successfully for $DATE."
    else
        echo "Download failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Starting data processing for $DATE..."
    python3 process_maroc.py --start-date "$DATE" --end-date "$DATE" --raw-data ./data/raw_rt/

    # Check if processing was successful
    if [ $? -eq 0 ]; then
        echo "Processing completed successfully for $DATE."
    else
        echo "Processing failed for $DATE. Skipping to next date."
        continue
    fi

    echo "Removing raw data for $DATE..."
    # Use quotes around the path to handle potential spaces in the path
    rm -rf "${DATA_RAW_PATH2}/$DATE"

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
python3 process_maid.py

echo "Completed pipeline for POI"
echo "----------------------------------------"

echo "Merging results"
python3 merge.py

echo "Completed pipeline for merging results"
echo "----------------------------------------"

echo "Pipeline processing completed."