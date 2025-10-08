import pygeohash as pgh
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
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

# Get blacklist file path from environment variable if available, otherwise use default
blacklist_file = os.environ.get('BLACKLIST_FILE', './blacklist_geohash.parquet')
black_list = pd.read_parquet(blacklist_file).gh7

# Define helper functions
def encode_geohash_batch(args):
    """Encode a batch of lat/lon to geohash with given precision.
    args is a tuple: (lat_list, lon_list, precision)
    Defined at module level to be picklable by multiprocessing.
    """
    lat_list, lon_list, precision = args
    return [pgh.encode(latitude=lat, longitude=lon, precision=precision) for lat, lon in zip(lat_list, lon_list)]

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
        rows = prepared_data[['geohash', 'timestamp', 'latitude', 'longitude', 'flux']].values.tolist()
        
        # Create store for this maid
        store = EvidenceStore(maid=maid)
        
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
        flux_values: Dict[str, List[str]] = {}
        
        # Process coordinates and flux first
        for gh, ts, lat, lon, flux in rows:
            if gh not in coordinates:
                coordinates[gh] = (lat, lon)
                flux_values[gh] = [flux] if flux is not None else []
            else:
                # Average the coordinates if we have multiple points
                current_lat, current_lon = coordinates[gh]
                coordinates[gh] = ((current_lat + lat) / 2, (current_lon + lon) / 2)
                # Collect flux values
                if flux is not None:
                    flux_values.setdefault(gh, []).append(flux)
        
        # Process timestamps separately
        for gh, ts, _, _, _ in rows:
            setin.setdefault(gh, []).append(ts)
        
        # Update store with timestamp data, coordinates, and flux values
        store.update(setin, coordinates, flux_values)
        
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

def process_dataset(raw_data_base, skip_existing_maids, tf_instance, output_dir, maid_mapping, valid_maids, start_date=None, end_date=None):
    """Process all data with optional date range filtering - tf_instance parameter kept for compatibility but not used"""
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
        # Set timezone to UTC to ensure consistent timestamp handling
        conn.execute("SET timezone='UTC'")
        # Configure DuckDB threads from environment or default to CPU count
        try:
            duckdb_threads = int(os.environ.get('DUCKDB_THREADS', str(mp.cpu_count())))
        except Exception:
            duckdb_threads = mp.cpu_count()
        try:
            conn.execute(f"PRAGMA threads={duckdb_threads}")
        except Exception:
            pass
        
        # Get all parquet files for this specific date folder
        folder_path = os.path.join(raw_data_base, date_folder)
        parquet_pattern = os.path.join(folder_path, '*.parquet')
        # Sort files to ensure deterministic input order across runs
        parquet_files = sorted(glob.glob(parquet_pattern))
        
        if not parquet_files:
            print(f"No parquet files found in {date_folder}, skipping...")
            conn.close()
            continue
        
        print(f"Found {len(parquet_files)} parquet files in {date_folder}")
        
        # Query data with deterministic behavior:
        # - Pre-filter to valid maids inside SQL via an in-memory table
        # - Add ORDER BY so DISTINCT ON picks the same row consistently
        print(f"Querying data filter for {date_folder}...")
        valid_maids_df = pd.DataFrame({'maid': list(valid_maids)})
        conn.register('valid_maids', valid_maids_df)
        data = conn.execute("""
            SELECT DISTINCT ON (latitude, longitude)
                   r.maid, r.timestamp, r.country, r.latitude, r.longitude, r.flux
            FROM read_parquet(?) AS r
            JOIN valid_maids AS v ON r.maid = v.maid
            ORDER BY r.latitude, r.longitude, r.maid, r.timestamp
        """, [parquet_files]).df()
        
        print(f"Loaded {len(data)} total rows from {date_folder}")
        
        if len(data) == 0:
            print(f"No records found in {date_folder}, skipping...")
            conn.close()
            continue
        
        
        if len(data) == 0:
            print(f"No valid MAIDs found in {date_folder}, skipping...")
            conn.close()
            continue
        
        # Map all MAIDs to canonical form (column 0)
        print(f"Mapping MAIDs to canonical form...")
        #data['maid'] = data['maid'].map(maid_mapping)
        print(f"Mapped {len(data['maid'].unique())} unique canonical MAIDs")
            
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
        
        # Convert timestamp to datetime
        print(f"Converting timestamps for {len(data)} records...")

        # Skip data cleaning step since data is already in UTC
        cleaned_data = data
        
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
        
        # Parallel geohash encoding with batches across processes
        print("Encoding geohashes in parallel...")
        try:
            gh_precision = int(os.environ.get('GEOHASH_PRECISION', '7'))
        except Exception:
            gh_precision = 7
        try:
            gh_batch_size = int(os.environ.get('GEOHASH_BATCH_SIZE', '500000'))
        except Exception:
            gh_batch_size = 500000
        try:
            gh_workers_env = int(os.environ.get('GEOHASH_WORKERS', '0'))
        except Exception:
            gh_workers_env = 0
        cpu_count = mp.cpu_count()
        if gh_workers_env and gh_workers_env > 0:
            gh_workers = min(gh_workers_env, cpu_count)
        else:
            gh_workers = max(1, min(cpu_count, 96))
        try:
            gh_chunksize = int(os.environ.get('GEOHASH_MAP_CHUNKSIZE', '10'))
        except Exception:
            gh_chunksize = 10

        total_rows = len(cleaned_data)
        geohashes = []

        # Prepare batch tasks
        batch_args = []
        for i in range(0, total_rows, gh_batch_size):
            batch_end = min(i + gh_batch_size, total_rows)
            print(f"Geohash batch {i//gh_batch_size + 1}/{(total_rows-1)//gh_batch_size + 1} ({batch_end - i} rows)")
            # Slice as Python lists for cheaper pickling
            batch_lat = cleaned_data['latitude'].iloc[i:batch_end].tolist()
            batch_lon = cleaned_data['longitude'].iloc[i:batch_end].tolist()
            batch_args.append((batch_lat, batch_lon, gh_precision))

        if len(batch_args) == 1:
            # Single batch: avoid process pool overhead
            geohashes = encode_geohash_batch(batch_args[0])
        else:
            with ProcessPoolExecutor(max_workers=gh_workers) as gh_executor:
                for result in tqdm(
                    gh_executor.map(encode_geohash_batch, batch_args, chunksize=gh_chunksize),
                    total=len(batch_args),
                    desc=f"Geohash encoding"
                ):
                    geohashes.extend(result)
        
        # Add geohashes to dataframe
        cleaned_data['geohash'] = geohashes
        cleaned_data=cleaned_data[~cleaned_data['geohash'].isin(black_list)]
        # Group cleaned data by maid
        grouped_data = cleaned_data.groupby('maid')
        
        # Prepare data for parallel processing
        maid_data_list = [(maid, group) for maid, group in grouped_data]
        
        print(f"Found {len(maid_data_list)} unique MAIDs to process in {date_folder}")
        
        # Use ProcessPoolExecutor to process all maids with progress bar
        # Limit number of processes to avoid memory issues and allow env override
        try:
            max_workers_env = int(os.environ.get('MAX_WORKERS', '0'))
        except Exception:
            max_workers_env = 0
        cpu_count = mp.cpu_count()
        if max_workers_env and max_workers_env > 0:
            num_processes = min(len(maid_data_list), max_workers_env)
        else:
            # Default to a conservative fraction to avoid I/O contention
            num_processes = min(len(maid_data_list), max(1, min(cpu_count, 96)))

        # Chunksize for executor.map to reduce overhead
        try:
            map_chunksize = int(os.environ.get('MAP_CHUNKSIZE', '100'))
        except Exception:
            map_chunksize = 100
        
        # Set output_dir in environment for child processes
        os.environ['PROCESS_OUTPUT_DIR'] = output_dir
            
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(
                executor.map(process_with_output_dir, maid_data_list, chunksize=map_chunksize), 
                total=len(maid_data_list), 
                desc=f"Processing MAIDs for {date_folder}"
            ))
        
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
    # Load MAID tuple mapping for deduplication (only once in main process)
    maid_tuple_file = os.environ.get('MAID_TUPLE_FILE', './maid_tuple.feather')
    print(f"Loading MAID tuple mapping from {maid_tuple_file}...")
    maid_tuple = pd.read_feather(maid_tuple_file)
    maid_mapping = {}
    for _, row in maid_tuple.iterrows():
        canonical_maid = row[0]
        duplicate_maid = row[1]
        maid_mapping[canonical_maid] = canonical_maid
        maid_mapping[duplicate_maid] = canonical_maid
    valid_maids = set(maid_mapping.keys())
    print(f"Loaded {len(maid_tuple)} MAID pairs, total {len(valid_maids)} valid MAIDs")
    
    # Constants
    skip_existing_maids = False  # Set to False to reprocess existing MAIDs

    # Parse command line arguments
    start_date = None
    end_date = None
    raw_data_base = os.environ.get('DATA_RAW_PATH', './data/raw_rt')  # Use env var or default to relative path
    output_dir = os.environ.get('OUTPUT_DIR', './data/processed_2')  # Use env var or default to relative path

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

    # Process all data with date range - removed timezone finder parameter
    date_range_info = f" from {start_date.strftime('%Y-%m-%d') if start_date else 'beginning'} to {end_date.strftime('%Y-%m-%d') if end_date else 'end'}"
    print(f"Processing all data with date range:{date_range_info}")
    process_dataset(raw_data_base, skip_existing_maids, None, output_dir, maid_mapping, valid_maids, start_date, end_date)

    print("\n" + "="*60)
    print("Completed processing all data")
    print("="*60)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main()