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
        "2025-03-023"
        "2025-03-24"
        "2025-03-25"
        "2025-03-26"
        "2025-03-27"
        "2025-03-28"
        "2025-03-29"
        "2025-03-30"
        "2025-03-31"
        "2025-04-01"
        "2025-04-02"
        "2025-04-03"
        "2025-04-04"
        "2025-04-05"
        "2025-04-06"
        "2025-04-07"
        "2025-04-08"
        "2025-04-09"
        "2025-04-10"
        "2025-04-11"
        "2025-04-12"
        "2025-04-13"
        "2025-04-14"
        "2025-04-15"
        "2025-04-16"
        "2025-04-17"
        "2025-04-18"
        "2025-04-19"
        "2025-04-20"
        "2025-04-21"
        "2025-04-22"
        "2025-04-23"
        "2025-04-24"
        "2025-04-25"
        "2025-04-26"
        "2025-04-27"
        "2025-04-28"
        "2025-04-29"
        "2025-04-30"
        "2025-05-01"
        "2025-05-02"
        "2025-05-03"
        "2025-05-04"
        "2025-05-05"
        "2025-05-06"
        "2025-05-07"
        "2025-05-08"
        "2025-05-09"
        "2025-05-10"
        "2025-05-11"
        "2025-05-12"
        "2025-05-13"
        "2025-05-14"
        "2025-05-15"
        "2025-05-16"
        "2025-05-17"
        "2025-05-18"
        "2025-05-19"
        "2025-05-20"
        "2025-05-21"
        "2025-05-22"
        "2025-05-23"
        "2025-05-24"
        "2025-05-25"
        "2025-05-26"
        "2025-05-27"
        "2025-05-28"
        "2025-05-29"
        "2025-05-30"
        "2025-05-31"
        "2025-06-01"
        "2025-06-02"
        "2025-06-03"
        "2025-06-04"
        "2025-06-05"
        "2025-06-06"
        "2025-06-07"
        "2025-06-08"
        "2025-06-09"
        "2025-06-10"
        "2025-06-11"
        "2025-06-12"
        "2025-06-13"
        "2025-06-14"
        "2025-06-15"
        "2025-06-16"
        "2025-06-17"
        "2025-06-18"
        "2025-06-19"
        "2025-06-20"
        "2025-06-21"
        "2025-06-22"
        "2025-06-23"
        "2025-06-24"
        "2025-06-25"
        "2025-06-26"
        "2025-06-27"
        "2025-06-28"
        "2025-06-29"
        "2025-06-30"
        "2025-07-01"
        "2025-07-02"
        "2025-07-03"
        "2025-07-04"
        "2025-07-05"
        "2025-07-06"
        "2025-07-07"
        "2025-07-08"
        "2025-07-09"
        "2025-07-10"
        "2025-07-11"
        "2025-07-12"
        "2025-07-13"
        "2025-07-14"
        "2025-07-15"
        "2025-07-16"
        "2025-07-17"
        "2025-07-18"
        "2025-07-19"
        "2025-07-20"
        "2025-07-21"
        "2025-07-22"
        "2025-07-23"
        "2025-07-24"
        "2025-07-25"
        "2025-07-26"
        "2025-07-27"
        "2025-07-28"
        "2025-07-29"
        "2025-07-30"
    )
    DATES2=(
        "2025-08-05"
        "2025-08-06"
        "2025-08-07"
        "2025-08-08"
        "2025-08-09"
        "2025-08-10"
        "2025-08-11"
        "2025-08-12"
        "2025-08-13"
        "2025-08-14"
        "2025-08-15"
        "2025-08-16"
        "2025-08-17"
        "2025-08-18"
        "2025-08-19"
        "2025-08-20"
        "2025-08-21"
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