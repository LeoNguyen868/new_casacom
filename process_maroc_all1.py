import pygeohash as pgh
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import os
import re
import glob
from envidence_new import EvidenceStore
from collections import defaultdict
import duckdb
import sys
import gc
from datetime import datetime

# Helper function for environment variable handling
def get_env_int(key, default):
    """Get integer value from environment variable with fallback"""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default

def get_env_str(key, default):
    """Get string value from environment variable with fallback"""
    return os.environ.get(key, default)

def cleanup_dataframe(df, name=""):
    """Helper function to cleanup DataFrame and free memory"""
    if df is not None:
        del df
        gc.collect()
        if name:
            print(f"Cleaned up {name} DataFrame")
    return None

# Get blacklist file path from environment variable if available, otherwise use default
blacklist_file = get_env_str('BLACKLIST_FILE', './blacklist_geohash.parquet')
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
        prepared_data.sort_values(by='timestamp', inplace=True)
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
        
        # Group data by geohash
        timestamp_data = defaultdict(list)
        coordinates = defaultdict(list)
        flux_data = defaultdict(list)

        for gh, ts, lat, lon, flux in rows:
            timestamp_data[gh].append(ts)
            coordinates[gh].append(pgh.encode(lat, lon, precision=12))
            if flux is not None:
                flux_data[gh].append(flux)
        store.update(timestamp_data, coordinates, flux_data)
        
        # Save store to pickle file
        try:
            store.save(full_path, compress=False)
        except Exception as e:
            print(f"Error saving store to pickle file: {e}")
            pass
        
        # No need to return store data, the function just saves it
        return maid, True
        
    except Exception:
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
    """Process all data with optional date range filtering - processes in file batches to maximize RAM usage"""
    print(f"\n{'='*60}")
    print(f"Processing all data in file batches (optimized for 96GB RAM)")
    if start_date or end_date:
        date_parts = []
        if start_date:
            date_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
        else:
            date_parts.append("from beginning")
        if end_date:
            date_parts.append(f"to {end_date.strftime('%Y-%m-%d')}")
        else:
            date_parts.append("to end")
        date_range_str = " ".join(date_parts)
        print(f"Date range: {date_range_str}")
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

    # Collect all parquet files from all date folders
    print(f"\nCollecting parquet files from all date folders...")
    all_parquet_files = [
        f for date_folder in date_folders
        for f in sorted(glob.glob(os.path.join(raw_data_base, date_folder, '*.parquet')))
    ]
    
    if not all_parquet_files:
        print(f"No parquet files found in any date folder, exiting...")
        return
    
    print(f"Found total {len(all_parquet_files)} parquet files across all dates")
    
    file_batch_size = get_env_int('FILE_BATCH_SIZE', 3000)
    
    print(f"File batch size: {file_batch_size} files (DuckDB will query this many files at once)")
    
    # Create single DuckDB connection for all processing
    conn = duckdb.connect()
    conn.execute("SET timezone='UTC'")
    
    # Configure DuckDB for optimal parallel processing
    duckdb_threads = get_env_int('DUCKDB_THREADS', mp.cpu_count())

    # Set DuckDB configuration for optimal parquet reading and parallel processing
    try:
        conn.execute(f"PRAGMA threads={duckdb_threads}")
        conn.execute("PRAGMA memory_limit='80GB'")
        print(f"DuckDB threads: {duckdb_threads}")
        print(f"DuckDB memory limit: 80GB")
        print(f"Object cache: enabled")
    except Exception as e:
        print(f"Warning: Could not set all DuckDB optimizations: {e}")
        pass
    
    # Process files in large batches to maximize RAM usage
    total_batches = (len(all_parquet_files) - 1) // file_batch_size + 1
    
    for batch_idx in range(0, len(all_parquet_files), file_batch_size):
        batch_files = all_parquet_files[batch_idx:batch_idx + file_batch_size]
        current_batch_num = batch_idx // file_batch_size + 1
        
        print(f"\n{'='*60}")
        print(f"Processing file batch {current_batch_num}/{total_batches}")
        print(f"Files in this batch: {len(batch_files)}")
        print(f"{'='*60}")
        
        # Query all files in this batch at once with DuckDB
        print(f"Querying {len(batch_files)} parquet files with DuckDB...")
        
        file_list = "', '".join(batch_files)
        # Register valid_maids as a DuckDB table for efficient filtering
        conn.register('valid_maids_table', pd.DataFrame({'maid': list(valid_maids)}))
        
        data = conn.execute(f"""
            SELECT p.maid, p.timestamp, p.latitude, p.longitude, p.flux
            FROM read_parquet(['{file_list}']) p
            INNER JOIN valid_maids_table v ON p.maid = v.maid
            WHERE p.latitude BETWEEN -90 AND 90 
            AND p.longitude BETWEEN -180 AND 180
        """).df()
        
        print(f"Loaded {len(data):,} total rows in batch {current_batch_num}/{total_batches}")
        
        if len(data) == 0:
            print(f"No records found in this batch, skipping...")
            continue
        
        # Map all MAIDs to canonical form
        print(f"Mapping MAIDs to canonical form...")
        original_unique_maids = len(data['maid'].unique())
        print(f"  Original unique MAIDs: {original_unique_maids:,}")
        data['maid'] = data['maid'].map(maid_mapping).fillna(data['maid'])
        mapped_unique_maids = len(data['maid'].unique())
        print(f"  Mapped unique MAIDs: {mapped_unique_maids:,}")
            
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
                print(f"After filtering existing files: {len(maids_to_process):,} unique MAIDs remaining")
            else:
                print(f"All MAIDs already processed for this batch, skipping...")
                del data
                gc.collect()
                continue
        
        # Data is already in UTC, no need to clean
        cleaned_data = data
        data = cleanup_dataframe(data, "raw_batch_data")
        
        # Parallel geohash encoding
        print(f"Encoding geohashes for {len(cleaned_data):,} records...")
        
        gh_precision = get_env_int('GEOHASH_PRECISION', 7)
        gh_batch_size = get_env_int('GEOHASH_BATCH_SIZE', 500000)
        gh_workers_env = get_env_int('GEOHASH_WORKERS', 0)
        cpu_count = mp.cpu_count()
        if gh_workers_env and gh_workers_env > 0:
            gh_workers = min(gh_workers_env, cpu_count)
        else:
            gh_workers = max(1, min(cpu_count, 96))
        gh_chunksize = get_env_int('GEOHASH_MAP_CHUNKSIZE', 10)

        total_rows = len(cleaned_data)
        geohashes = []

        # Prepare batch tasks
        batch_args = []
        for i in range(0, total_rows, gh_batch_size):
            batch_end = min(i + gh_batch_size, total_rows)
            # Slice as Python lists for cheaper pickling
            batch_lat = cleaned_data['latitude'].iloc[i:batch_end].tolist()
            batch_lon = cleaned_data['longitude'].iloc[i:batch_end].tolist()
            batch_args.append((batch_lat, batch_lon, gh_precision))

        if len(batch_args) == 1:
            # Single batch: avoid process pool overhead
            print(f"  Single geohash batch, processing directly...")
            geohashes = encode_geohash_batch(batch_args[0])
        else:
            print(f"  Processing {len(batch_args)} geohash batches with {gh_workers} workers...")
            with ProcessPoolExecutor(max_workers=gh_workers) as gh_executor:
                for result in tqdm(
                    gh_executor.map(encode_geohash_batch, batch_args, chunksize=gh_chunksize),
                    total=len(batch_args),
                    desc=f"  Geohash encoding (batch {current_batch_num}/{total_batches})"
                ):
                    geohashes.extend(result)
        
        # Add geohashes to dataframe
        cleaned_data['geohash'] = geohashes
        del geohashes
        gc.collect()
        
        # Prepare data for parallel processing - optimized groupby
        print(f"Grouping data by MAID...")
        maid_data_list = list(cleaned_data.groupby('maid', sort=False))
        cleaned_data = cleanup_dataframe(cleaned_data, "geohash_data")
        
        print(f"Found {len(maid_data_list):,} unique MAIDs to process in this batch")
        
        # Use ProcessPoolExecutor to process all maids with progress bar
        max_workers_env = get_env_int('MAX_WORKERS', 0)
        cpu_count = mp.cpu_count()
        if max_workers_env and max_workers_env > 0:
            num_processes = min(len(maid_data_list), max_workers_env)
        else:
            num_processes = min(len(maid_data_list), max(1, min(cpu_count, 96)))

        map_chunksize = get_env_int('MAP_CHUNKSIZE', 250)
        
        print(f"Processing MAIDs with {num_processes} workers... map_chunksize: {map_chunksize}")
        
        # Set output_dir in environment for child processes
        os.environ['PROCESS_OUTPUT_DIR'] = output_dir
            
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(
                executor.map(process_with_output_dir, maid_data_list, chunksize=map_chunksize), 
                total=len(maid_data_list), 
                desc=f"  Processing MAIDs (batch {current_batch_num}/{total_batches})"
            ))
        
        # Count successful results
        successful_maids = sum(1 for _, success in results if success)
        
        print(f"Completed processing {successful_maids:,}/{len(maid_data_list):,} MAIDs from batch {current_batch_num}/{total_batches}")
        
        # Collect garbage after processing each batch to free up memory
        del maid_data_list
        del results
        gc.collect()  # Keep explicit gc.collect() for batch-level cleanup

    # Close DuckDB connection
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"Completed processing all file batches")
    print(f"{'='*60}")

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
    import pickle
    import glob
    tmps=glob.glob('./data/processed_all/*.tmp')
    cc=set([pickle.load(open(tmps[i],'rb'))['maid'] for i in range(len(tmps))])
    # Load MAID tuple mapping for deduplication (only once in main process)
    maid_tuple_file = get_env_str('MAID_MAPPING_FILE', './maid_mapping.pkl')
    print(f"Loading MAID tuple mapping from {maid_tuple_file}...")
    maid_mapping = pickle.load(open(maid_tuple_file, 'rb'))
    valid_maids = set(maid_mapping.keys())
    print(f"Loaded {len(maid_mapping):,} MAID pairs, total {len(valid_maids):,} valid MAIDs")
    
    # Constants
    skip_existing_maids = False  # Set to False to reprocess existing MAIDs

    # Parse command line arguments
    start_date = None
    end_date = None
    raw_data_base = get_env_str('DATA_RAW_PATH', './data/raw_rt')  # Use env var or default to relative path
    output_dir = get_env_str('OUTPUT_DIR', './data/processed_all')  # Use env var or default to relative path

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
    date_parts = []
    if start_date:
        date_parts.append(f"from {start_date.strftime('%Y-%m-%d')}")
    else:
        date_parts.append("from beginning")
    if end_date:
        date_parts.append(f"to {end_date.strftime('%Y-%m-%d')}")
    else:
        date_parts.append("to end")
    date_range_info = " ".join(date_parts)
    print(f"Processing all data with date range: {date_range_info}")
    process_dataset(raw_data_base, skip_existing_maids, None, output_dir, maid_mapping, valid_maids, start_date, end_date)

    print("\n" + "="*60)
    print("Completed processing all data")
    print("="*60)

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)

    # Set optimal environment variables for DuckDB performance if not already set
    if not os.environ.get('DUCKDB_THREADS'):
        os.environ['DUCKDB_THREADS'] = str(mp.cpu_count())
        print(f"Setting DUCKDB_THREADS to {mp.cpu_count()}")

    if not os.environ.get('FILE_BATCH_SIZE'):
        os.environ['FILE_BATCH_SIZE'] = '300'
        print("Setting FILE_BATCH_SIZE to 3000 for optimal performance")

    main()
