import pygeohash as pgh
import timezonefinder as tf
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import logging
from timezonefinder import TimezoneFinder
import json
import os
import re
import glob
import pandas as pd
from envidence import EvidenceStore
from typing import Dict, List, Tuple
import duckdb
import sys
import gc
from datetime import datetime

tf = TimezoneFinder()
# Get blacklist file path from environment variable if available, otherwise use default
blacklist_file = os.environ.get('BLACKLIST_FILE', './blacklist_geohash.parquet')
black_list = pd.read_parquet(blacklist_file).gh7

# Define helper functions
def clean_maid_dat(maid_dat):
    if len(maid_dat) == 0:
        return maid_dat
    
    # Create a copy of the dataframe to avoid modifying the original
    maid_dat = maid_dat.copy()
    
    # Localize naive timestamps to UTC
    if maid_dat['timestamp'].dt.tz is None:
        maid_dat['timestamp'] = maid_dat['timestamp'].dt.tz_localize('UTC')
    # Convert timestamps to local timezone
    local_tz = tf.timezone_at(lat=maid_dat['latitude'].values[0], lng=maid_dat['longitude'].values[0])
    maid_dat['timestamp'] = maid_dat['timestamp'].dt.tz_convert(local_tz)
    return maid_dat

def make_safe_filename(maid):
    """Convert maid to safe filename"""
    # Replace unsafe characters with underscores
    safe_name = re.sub(r'[^\w\-_.]', '_', str(maid))
    return f"{safe_name}.pkl"

def get_existing_maid_files(output_dir):
    """Get set of MAIDs that already have processed files"""
    existing_files = glob.glob(os.path.join(output_dir, '*.pkl'))
    existing_maids = set()
    
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # Remove .pkl extension to get the safe filename
        safe_name = filename[:-4]  # Remove .pkl
        existing_maids.add(safe_name)
    
    return existing_maids

def process_maid_data(maid_data, output_dir):
    """Process data for a single maid - simplified to focus on store operations"""
    maid, prepared_data = maid_data
    
    try:
        # Create safe filename for this maid
        safe_filename = make_safe_filename(maid)
        full_path = os.path.join(output_dir, safe_filename)
        
        # Convert pandas DataFrame to numpy array to avoid serialization issues
        # Get rows containing geohash, timestamp, latitude, and longitude
        rows = prepared_data[['geohash', 'timestamp', 'latitude', 'longitude']].values.tolist()
        
        # Create store for this maid
        store = EvidenceStore()
        
        # Try to load existing data from pickle file
        if os.path.exists(full_path):
            try:
                store.load(full_path)
            except Exception:
                # If loading fails, continue with empty store
                pass
        
        # Prepare timestamp data and coordinates - group by geohash
        setin: Dict[str, List[str]] = {}
        coordinates: Dict[str, Tuple[float, float]] = {}
        
        # Process coordinates first
        for gh, ts, lat, lon in rows:
            if gh not in coordinates:
                coordinates[gh] = (lat, lon)
            else:
                # Average the coordinates if we have multiple points
                current_lat, current_lon = coordinates[gh]
                coordinates[gh] = ((current_lat + lat) / 2, (current_lon + lon) / 2)
        
        # Process timestamps separately
        for gh, ts, _, _ in rows:
            setin.setdefault(gh, []).append(ts)
        
        # Update store with timestamp data and coordinates
        store.update(setin, coordinates)
        
        # Save store to pickle file
        try:
            store.save(full_path, compress=False)
        except Exception as e:
            print(f"Error saving store to pickle file: {e}")
            pass
        
        # No need to return store data, the function just saves it
        return maid, True
        
    except Exception as e:
        # Suppress error messages to avoid cluttering output during multiprocessing
        return maid, False

def filter_date_folders(date_folders, start_date=None, end_date=None):
    """Filter date folders based on start and end date range"""
    if start_date is None and end_date is None:
        return date_folders
    
    filtered_folders = []
    for date_folder in date_folders:
        try:
            folder_date = datetime.strptime(date_folder, '%Y-%m-%d')
            
            # Check if folder date is within the specified range
            if start_date and folder_date < start_date:
                continue
            if end_date and folder_date > end_date:
                continue
                
            filtered_folders.append(date_folder)
        except ValueError:
            # Skip folders that don't match the date format
            continue
    
    return filtered_folders

def process_dataset(raw_data_base, skip_existing_maids, tf_instance, output_dir, start_date=None, end_date=None):
    """Process all data with optional date range filtering"""
    print(f"\n{'='*60}")
    print(f"Processing all data")
    if start_date or end_date:
        date_range_str = f"Date range: {start_date.strftime('%Y-%m-%d') if start_date else 'beginning'} to {end_date.strftime('%Y-%m-%d') if end_date else 'end'}"
        print(f"{date_range_str}")
    print(f"{'='*60}")
    
    # Get existing processed MAIDs to skip (if enabled)
    if skip_existing_maids:
        print(f"Checking for existing processed MAID files...")
        existing_maid_files = get_existing_maid_files(output_dir)
        print(f"Found {len(existing_maid_files)} existing MAID files that will be skipped")
    else:
        print(f"Skip existing MAIDs is disabled - will reprocess all MAIDs")
        existing_maid_files = set()

    # Find all date folders in raw data directory
    all_date_folders = [d for d in os.listdir(raw_data_base) 
                        if os.path.isdir(os.path.join(raw_data_base, d)) and 
                        re.match(r'\d{4}-\d{2}-\d{2}', d)]
    all_date_folders.sort()
    
    # Filter date folders based on date range
    date_folders = filter_date_folders(all_date_folders, start_date, end_date)

    print(f"Found {len(date_folders)} date folders to process: {date_folders}")
    if start_date or end_date:
        print(f"(Filtered from {len(all_date_folders)} total date folders based on date range)")

    # Process each date folder one by one
    for date_folder in date_folders:
        print(f"\nProcessing date folder: {date_folder}")
        
        # Create DuckDB connection for this date
        conn = duckdb.connect()
        
        # Get all parquet files for this specific date folder
        folder_path = os.path.join(raw_data_base, date_folder)
        parquet_pattern = os.path.join(folder_path, '*.parquet')
        parquet_files = glob.glob(parquet_pattern)
        
        if not parquet_files:
            print(f"No parquet files found in {date_folder}, skipping...")
            conn.close()
            continue
        
        print(f"Found {len(parquet_files)} parquet files in {date_folder}")
        
        # Query data directly with just country='MAR' filter
        print(f"Querying data filter for {date_folder}...")
        data = conn.execute("""
            SELECT DISTINCT ON (latitude, longitude) maid, timestamp, country, latitude, longitude 
            FROM read_parquet(?)
            ORDER BY maid, timestamp
        """, [parquet_files]).df()
        
        print(f"Loaded {len(data)} total rows from {date_folder}")
        
        if len(data) == 0:
            print(f"No records found in {date_folder}, skipping...")
            conn.close()
            continue
            
        # Filter out MAIDs that already have processed files (if skip_existing_maids is enabled)
        if skip_existing_maids:
            unique_maids = data['maid'].unique()
            maids_to_process = []
            
            for maid in unique_maids:
                safe_filename = make_safe_filename(maid)
                safe_name = safe_filename[:-4]  # Remove .pkl extension
                if safe_name not in existing_maid_files:
                    maids_to_process.append(maid)
                    
            # Filter data to only include MAIDs that need processing
            if len(maids_to_process) > 0:
                data = data[data['maid'].isin(maids_to_process)]
                print(f"After filtering existing files: {len(maids_to_process)} unique MAIDs remaining to process in {date_folder}")
            else:
                print(f"All MAIDs already processed for {date_folder}, skipping...")
                conn.close()
                continue
        
        # Clean data and add geohash before threading (like in script_HCM.py)
        print(f"Cleaning data and adding geohash for {len(data)} records...")
        
        # Apply clean_maid_dat to the entire dataset grouped by MAID
        # First convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Clean data by MAID group
        cleaned_data_list = []
        for maid, group in tqdm(data.groupby('maid'), desc=f"Cleaning data for {date_folder}"):
            try:
                cleaned_group = clean_maid_dat(group)
                cleaned_data_list.append(cleaned_group)
            except Exception as e:
                # Skip problematic groups
                continue
                
        # Combine all cleaned groups
        if not cleaned_data_list:
            print(f"No valid data after cleaning for {date_folder}, skipping...")
            conn.close()
            continue
            
        cleaned_data = pd.concat(cleaned_data_list)
        
        # Add geohash column using vectorized operation for all data at once
        print(f"Adding geohash to {len(cleaned_data)} records...")
        
        # Filter out invalid coordinates (NaN, inf, out of bounds)
        valid_coords = ((cleaned_data['latitude'] >= -90.0) & 
                        (cleaned_data['latitude'] <= 90.0) & 
                        (cleaned_data['longitude'] >= -180.0) & 
                        (cleaned_data['longitude'] <= 180.0) &
                        (cleaned_data['latitude'].notna()) &
                        (cleaned_data['longitude'].notna()) &
                        (~cleaned_data['latitude'].isin([float('inf'), float('-inf')]) &
                        (~cleaned_data['longitude'].isin([float('inf'), float('-inf')]))))
        
        invalid_count = len(cleaned_data) - valid_coords.sum()
        if invalid_count > 0:
            print(f"WARNING: Found {invalid_count} records with invalid coordinates that will be filtered out")
            cleaned_data = cleaned_data[valid_coords]
            print(f"Continuing with {len(cleaned_data)} valid records...")
        
        # Use vectorized operations instead of apply for better performance
        print("Encoding geohashes with vectorized operations...")
        from functools import partial
        import numpy as np
        
        # Vectorize the geohash encoding function
        def encode_geohash_bulk(lat, lon, precision=7):
            return pgh.encode(latitude=lat, longitude=lon, precision=precision)
        
        # Apply encoding in batches to reduce memory usage
        batch_size = 100000
        total_rows = len(cleaned_data)
        geohashes = []
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            print(f"Processing batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} ({batch_end - i} rows)")
            
            # Apply encoding to this batch
            batch_lat = cleaned_data['latitude'].iloc[i:batch_end].values
            batch_lon = cleaned_data['longitude'].iloc[i:batch_end].values
            
            # Using list comprehension for better performance than apply
            batch_geohashes = [encode_geohash_bulk(lat, lon) for lat, lon in zip(batch_lat, batch_lon)]
            geohashes.extend(batch_geohashes)
        
        # Add geohashes to dataframe
        cleaned_data['geohash'] = geohashes
        cleaned_data=cleaned_data[~cleaned_data['geohash'].isin(black_list)]
        # Group cleaned data by maid
        grouped_data = cleaned_data.groupby('maid')
        
        # Prepare data for parallel processing
        maid_data_list = [(maid, group) for maid, group in grouped_data]
        
        print(f"Found {len(maid_data_list)} unique MAIDs to process in {date_folder}")
        
        # Use ProcessPoolExecutor to process all maids with progress bar
        # Limit number of processes to avoid memory issues
        num_processes = min(len(maid_data_list), mp.cpu_count())  # Use CPU count as reference
        
        # Set output_dir in environment for child processes
        os.environ['PROCESS_OUTPUT_DIR'] = output_dir
            
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(executor.map(process_with_output_dir, maid_data_list), 
                               total=len(maid_data_list), 
                               desc=f"Processing MAIDs for {date_folder}"))
        
        # Count successful results
        successful_maids = sum(1 for _, success in results if success)
        
        print(f"Completed processing {successful_maids} MAIDs from {date_folder}")
        
        # Close DuckDB connection for this date
        conn.close()
        
        # Collect garbage after processing each date to free up memory
        gc.collect()

    print(f"\nCompleted processing all date folders")

def process_with_output_dir(maid_data):
    """Wrapper function for process_maid_data that can be used with ProcessPoolExecutor"""
    # Extract output_dir from environment variable set before process pool creation
    output_dir = os.environ.get('PROCESS_OUTPUT_DIR', '')
    return process_maid_data(maid_data, output_dir)

def parse_date_argument(date_str):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")
        sys.exit(1)

def main():
    """Main function to process all datasets"""
    # Suppress timezonefinder error messages
    logging.getLogger('timezonefinder').setLevel(logging.CRITICAL)

    # Initialize timezone finder globally for threading
    tf_instance = TimezoneFinder()

    # Constants
    skip_existing_maids = False  # Set to False to reprocess existing MAIDs
    
    # Parse command line arguments
    start_date = None
    end_date = None
    raw_data_base = os.environ.get('DATA_RAW_PATH', './data/raw_rt')  # Use env var or default to relative path
    output_dir = os.environ.get('OUTPUT_DIR', './data/processed')  # Use env var or default to relative path
    
    # Parse arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--start-date' and i + 1 < len(sys.argv):
            start_date = parse_date_argument(sys.argv[i + 1])
            i += 1  # Skip next argument as it's the date value
        elif arg == '--end-date' and i + 1 < len(sys.argv):
            end_date = parse_date_argument(sys.argv[i + 1])
            i += 1  # Skip next argument as it's the date value
        elif arg == '--raw-data' and i + 1 < len(sys.argv):
            raw_data_base = sys.argv[i + 1]
            i += 1  # Skip next argument as it's the path value
        elif arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
            i += 1  # Skip next argument as it's the path value
        elif re.match(r'\d{4}-\d{2}-\d{2}', arg):
            # If it's a date format, treat as start date if not already set
            if start_date is None:
                start_date = parse_date_argument(arg)
            elif end_date is None:
                end_date = parse_date_argument(arg)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python process_maroc.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--raw-data PATH] [--output-dir PATH]")
            print("       python process_maroc.py [start_date] [end_date]")
            print("Examples:")
            print("  python process_maroc.py")
            print("  python process_maroc.py --start-date 2025-08-05")
            print("  python process_maroc.py --start-date 2025-08-05 --end-date 2025-08-10")
            print("  python process_maroc.py 2025-08-05 2025-08-10")
            print("  python process_maroc.py --raw-data /path/to/raw/data --output-dir /path/to/output")
            sys.exit(1)
        
        i += 1
    
    # Create directory for processed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all data with date range
    date_range_info = f" from {start_date.strftime('%Y-%m-%d') if start_date else 'beginning'} to {end_date.strftime('%Y-%m-%d') if end_date else 'end'}"
    print(f"Processing all data with date range:{date_range_info}")
    process_dataset(raw_data_base, skip_existing_maids, tf_instance, output_dir, start_date, end_date)
    
    print("\n" + "="*60)
    print("Completed processing all data")
    print("="*60)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main()