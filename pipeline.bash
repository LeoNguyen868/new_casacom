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
        "2025-03-23"
        "2025-03-24"
        "2025-03-25"
        # "2025-03-26"
        # "2025-03-27"
        # "2025-03-28"
        # "2025-03-29"
        # "2025-03-30"
        # "2025-03-31"
        # "2025-04-01"
        # "2025-04-02"
        # "2025-04-03"
        # "2025-04-04"
        # "2025-04-05"
        # "2025-04-06"
        # "2025-04-07"
        # "2025-04-08"
        # "2025-04-09"
        # "2025-04-10"
        # "2025-04-11"
        # "2025-04-12"
        # "2025-04-13"
        # "2025-04-14"
        # "2025-04-15"
        # "2025-04-16"
        # "2025-04-17"
        # "2025-04-18"
        # "2025-04-19"
        # "2025-04-20"
        # "2025-04-21"
        # "2025-04-22"
        # "2025-04-23"
        # "2025-04-24"
        # "2025-04-25"
        # "2025-04-26"
        # "2025-04-27"
        # "2025-04-28"
        # "2025-04-29"
        # "2025-04-30"
        # "2025-05-01"
        # "2025-05-02"
        # "2025-05-03"
        # "2025-05-04"
        # "2025-05-05"
        # "2025-05-06"
        # "2025-05-07"
        # "2025-05-08"
        # "2025-05-09"
        # "2025-05-10"
        # "2025-05-11"
        # "2025-05-12"
        # "2025-05-13"
        # "2025-05-14"
        # "2025-05-15"
        # "2025-05-16"
        # "2025-05-17"
        # "2025-05-18"
        # "2025-05-19"
        # "2025-05-20"
        # "2025-05-21"
        # "2025-05-22"
        # "2025-05-23"
        # "2025-05-24"
        # "2025-05-25"
        # "2025-05-26"
        # "2025-05-27"
        # "2025-05-28"
        # "2025-05-29"
        # "2025-05-30"
        # "2025-05-31"
        # "2025-06-01"
        # "2025-06-02"
        # "2025-06-03"
        # "2025-06-04"
        # "2025-06-05"
        # "2025-06-06"
        # "2025-06-07"
        # "2025-06-08"
        # "2025-06-09"
        # "2025-06-10"
        # "2025-06-11"
        # "2025-06-12"
        # "2025-06-13"
        # "2025-06-14"
        # "2025-06-15"
        # "2025-06-16"
        # "2025-06-17"
        # "2025-06-18"
        # "2025-06-19"
        # "2025-06-20"
        # "2025-06-21"
        # "2025-06-22"
        # "2025-06-23"
        # "2025-06-24"
        # "2025-06-25"
        # "2025-06-26"
        # "2025-06-27"
        # "2025-06-28"
        # "2025-06-29"
        # "2025-06-30"
        # "2025-07-01"
        # "2025-07-02"
        # "2025-07-03"
        # "2025-07-04"
        # "2025-07-05"
        # "2025-07-06"
        # "2025-07-07"
        # "2025-07-08"
        # "2025-07-09"
        # "2025-07-10"
        # "2025-07-11"
        # "2025-07-12"
        # "2025-07-13"
        # "2025-07-14"
        # "2025-07-15"
        # "2025-07-16"
        # "2025-07-17"
        # "2025-07-18"
        # "2025-07-19"
        # "2025-07-20"
        # "2025-07-21"
        # "2025-07-22"
        # "2025-07-23"
        # "2025-07-24"
        # "2025-07-25"
        # "2025-07-26"
        # "2025-07-27"
        # "2025-07-28"
        # "2025-07-29"
        # "2025-07-30"
    )
    DATES2=(
        # "2025-08-05"
        # "2025-08-06"
        # "2025-08-07"
        # "2025-08-08"
        # "2025-08-09"
        # "2025-08-10"
        # "2025-08-11"
        # "2025-08-12"
        # "2025-08-13"
        # "2025-08-14"
        # "2025-08-15"
        # "2025-08-16"
        # "2025-08-17"
        # "2025-08-18"
        # "2025-08-19"
        # "2025-08-20"
        # "2025-08-21"
        # "2025-08-22"
        # "2025-08-23"
        # "2025-08-24"
        # "2025-08-25"
        # "2025-08-26"
        # "2025-08-27"
        # "2025-08-28"
        # "2025-08-29"
        # "2025-08-30"
        # "2025-08-31"
        # "2025-09-01"
        # "2025-09-02"
        # "2025-09-03"
        # "2025-09-04"
        # "2025-09-05"
        # "2025-09-06"
        # "2025-09-07"
        # "2025-09-08"
        # "2025-09-09"
        # "2025-09-10"
        # "2025-09-11"
        # "2025-09-12"
        # "2025-09-13"
        # "2025-09-14"
        # "2025-09-15"
        # "2025-09-16"
        # "2025-09-17"
        # "2025-09-18"
    )
else
    # Use provided arguments for both DATES1 and DATES2
    DATES1=("$@")
    DATES2=("$@")
fi

# Process DATES1 first in batches of 10
BATCH_SIZE=10
batch_dates=()
downloaded_in_batch=()

for DATE in "${DATES1[@]}"; do
    # Validate date format (YYYY-MM-DD)
    if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid date format: $DATE. Please use YYYY-MM-DD format."
        continue
    fi

    echo "Starting data download for $DATE..."
    if python3 download_maroc.py --start-date "$DATE" --end-date "$DATE" --local-dir ./data/raw/ --bucket s3://home-work/hw/project2; then
        echo "Download completed successfully for $DATE."
        downloaded_in_batch+=("$DATE")
    else
        echo "Download failed for $DATE. Skipping this date."
    fi

    batch_dates+=("$DATE")

    if [ ${#batch_dates[@]} -ge $BATCH_SIZE ]; then
        if [ ${#downloaded_in_batch[@]} -gt 0 ]; then
            MIN_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | head -n1)
            MAX_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | tail -n1)
            echo "Starting data processing for batch: $MIN_DATE -> $MAX_DATE"
            if python3 process_maroc_mongo.py --raw-data ./data/raw/; then
                echo "Processing completed successfully for batch: $MIN_DATE -> $MAX_DATE"
                echo "Removing raw data for processed dates in batch..."
                for RD in "${downloaded_in_batch[@]}"; do
                    rm -rf "${DATA_RAW_PATH1}/$RD"
                done
            else
                echo "Processing failed for batch: $MIN_DATE -> $MAX_DATE"
            fi
        else
            echo "No successful downloads in this batch; skipping processing."
        fi
        # Reset batch trackers
        batch_dates=()
        downloaded_in_batch=()
        echo "----------------------------------------"
    fi
done

# Process any remaining dates in the last (partial) batch
if [ ${#batch_dates[@]} -gt 0 ]; then
    if [ ${#downloaded_in_batch[@]} -gt 0 ]; then
        MIN_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | head -n1)
        MAX_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | tail -n1)
        echo "Starting data processing for final batch: $MIN_DATE -> $MAX_DATE"
        if python3 process_maroc_mongo.py --raw-data ./data/raw/; then
            echo "Processing completed successfully for final batch: $MIN_DATE -> $MAX_DATE"
            echo "Removing raw data for processed dates in final batch..."
            for RD in "${downloaded_in_batch[@]}"; do
                rm -rf "${DATA_RAW_PATH1}/$RD"
            done
        else
            echo "Processing failed for final batch: $MIN_DATE -> $MAX_DATE"
        fi
    else
        echo "No successful downloads in final batch; skipping processing."
    fi
    echo "----------------------------------------"
fi

# Then process DATES2 in batches of 10
batch_dates=()
downloaded_in_batch=()

for DATE in "${DATES2[@]}"; do
    # Validate date format (YYYY-MM-DD)
    if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        echo "Invalid date format: $DATE. Please use YYYY-MM-DD format."
        continue
    fi

    echo "Starting data download for $DATE..."
    if python3 download_maroc.py --start-date "$DATE" --end-date "$DATE" --local-dir ./data/raw_rt/ --bucket s3://home-work/hw/project2_rt; then
        echo "Download completed successfully for $DATE."
        downloaded_in_batch+=("$DATE")
    else
        echo "Download failed for $DATE. Skipping this date."
    fi

    batch_dates+=("$DATE")

    if [ ${#batch_dates[@]} -ge $BATCH_SIZE ]; then
        if [ ${#downloaded_in_batch[@]} -gt 0 ]; then
            MIN_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | head -n1)
            MAX_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | tail -n1)
            echo "Starting data processing for batch: $MIN_DATE -> $MAX_DATE"
            if python3 process_maroc_mongo.py --raw-data ./data/raw_rt/; then
                echo "Processing completed successfully for batch: $MIN_DATE -> $MAX_DATE"
                echo "Removing raw data for processed dates in batch..."
                for RD in "${downloaded_in_batch[@]}"; do
                    rm -rf "${DATA_RAW_PATH2}/$RD"
                done
            else
                echo "Processing failed for batch: $MIN_DATE -> $MAX_DATE"
            fi
        else
            echo "No successful downloads in this batch; skipping processing."
        fi
        # Reset batch trackers
        batch_dates=()
        downloaded_in_batch=()
        echo "----------------------------------------"
    fi
done

# Process any remaining dates in the last (partial) batch for DATES2
if [ ${#batch_dates[@]} -gt 0 ]; then
    if [ ${#downloaded_in_batch[@]} -gt 0 ]; then
        MIN_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | head -n1)
        MAX_DATE=$(printf "%s\n" "${downloaded_in_batch[@]}" | sort | tail -n1)
        echo "Starting data processing for final batch: $MIN_DATE -> $MAX_DATE"
        if python3 process_maroc_mongo.py --raw-data ./data/raw_rt/; then
            echo "Processing completed successfully for final batch: $MIN_DATE -> $MAX_DATE"
            echo "Removing raw data for processed dates in final batch..."
            for RD in "${downloaded_in_batch[@]}"; do
                rm -rf "${DATA_RAW_PATH2}/$RD"
            done
        else
            echo "Processing failed for final batch: $MIN_DATE -> $MAX_DATE"
        fi
    else
        echo "No successful downloads in final batch; skipping processing."
    fi
    echo "----------------------------------------"
fi

echo "Processing POI"
python3 process_maid.py

echo "Completed pipeline for POI"
echo "----------------------------------------"

echo "Merging results"
python3 merge.py

echo "Completed pipeline for merging results"
echo "----------------------------------------"

echo "Compressing results..."
cd result/
tar -czf ../results.tar.gz .
cd ..
echo "Results compressed to results.tar.gz"
echo "----------------------------------------"


echo "Pipeline processing completed."