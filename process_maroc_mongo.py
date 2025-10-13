#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Maroc data processing script that saves directly to MongoDB instead of pickle files.
This approach provides better performance and scalability for large datasets.
"""

import pygeohash as pgh
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import os
import re
import glob
from envidence_mongo import EvidenceStore
from collections import defaultdict
import duckdb
import sys
import gc
from datetime import datetime
from pymongo import MongoClient, UpdateOne
import pickle

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

def batch_query_existing_maids(collection, maid_list, batch_size=None):
    """Query existing MAIDs in batches to avoid MongoDB document size limit

    Args:
        collection: MongoDB collection to query
        maid_list: List of MAIDs to check for existence
        batch_size: Maximum number of MAIDs per query batch (default: from env var)

    Returns:
        tuple: (existing_maids_set, existing_docs_dict)
    """
    if batch_size is None:
        batch_size = get_env_int('MAID_QUERY_BATCH_SIZE', 100000)

    existing_maids = set()
    existing_docs = {}

    total_maids = len(maid_list)
    print(f"Batch querying {total_maids:,} MAIDs in batches of {batch_size:,}")

    for i in range(0, total_maids, batch_size):
        batch_end = min(i + batch_size, total_maids)
        current_batch = maid_list[i:batch_end]

        try:
            # Query current batch
            batch_docs = list(collection.find(
                {'_id': {'$in': current_batch}},
                {'_id': 1}  # Only get _id for efficiency
            ))

            # Add to results
            for doc in batch_docs:
                existing_maids.add(doc['_id'])
                existing_docs[doc['_id']] = doc

            batch_progress = min(i + batch_size, total_maids)
            print(f"  Processed {batch_progress:,}/{total_maids:,} MAIDs")

        except Exception as e:
            print(f"Warning: Error querying batch {i//batch_size + 1}: {e}")
            # Continue with next batch even if current batch fails
            continue

    print(f"Found {len(existing_maids):,} existing MAIDs out of {total_maids:,} total")
    return existing_maids, existing_docs

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

def process_maid_data(maid_data, batch_existing_docs=None):
    """Process data for a single maid - return store for bulk MongoDB save"""
    maid, prepared_data = maid_data

    try:
        # Convert pandas DataFrame to numpy array to avoid serialization issues
        # Get rows containing geohash, timestamp, latitude, and longitude
        prepared_data.sort_values(by='timestamp', inplace=True)
        rows = prepared_data[['geohash', 'timestamp', 'latitude', 'longitude', 'flux']].values.tolist()

        # Group data by geohash
        timestamp_data = defaultdict(list)
        coordinates = defaultdict(list)
        flux_data = defaultdict(list)

        for gh, ts, lat, lon, flux in rows:
            # Convert timestamp to Unix timestamp if it's a datetime object
            if hasattr(ts, 'timestamp'):  # It's a datetime object
                ts_float = ts.timestamp()
            else:  # It's already a float or int
                ts_float = float(ts)

            timestamp_data[gh].append(ts_float)
            coordinates[gh].append(pgh.encode(lat, lon, precision=12))
            if flux is not None:
                flux_data[gh].append(flux)

        # Check if MAID exists in current batch's existing docs for incremental update
        if batch_existing_docs and maid in batch_existing_docs:
            # MAID exists, need to load existing store and update
            from pymongo import MongoClient

            try:
                # Create temporary MongoDB connection to load existing data
                mongo_host = os.getenv('MONGO_HOST', 'localhost')
                mongo_port = int(os.getenv('MONGO_PORT', '27017'))
                mongo_db = os.getenv('MONGO_DB', 'casacom')
                mongo_collection = os.getenv('MONGO_COLLECTION', 'maids')

                temp_client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
                temp_collection = temp_client[mongo_db][mongo_collection]

                # Load existing document from MongoDB
                existing_doc = temp_collection.find_one({'_id': maid})
                temp_client.close()

                if existing_doc:
                    # print(f"Loading existing store for MAID: {maid}")
                    # Create store and load existing data
                    store = EvidenceStore(maid=maid)
                    store.load_from_mongo(existing_doc)
                    # Update with new data
                    store.update(timestamp_data, coordinates, flux_data)
                else:
                    # Fallback: create new store if document not found
                    print(f"Warning: MAID {maid} in existing list but not found in DB, creating new store")
                    store = EvidenceStore(maid=maid)
                    store.update(timestamp_data, coordinates, flux_data)
            except Exception as e:
                print(f"Error loading existing store for MAID {maid}: {e}")
                # Fallback: create new store
                store = EvidenceStore(maid=maid)
                store.update(timestamp_data, coordinates, flux_data)
        else:
            # Create new store (new MAID or no existing data)
            store = EvidenceStore(maid=maid)
            store.update(timestamp_data, coordinates, flux_data)

        # Return store object for bulk MongoDB save
        return maid, store, True

    except Exception as e:
        print(f"Error processing maid {maid}: {e}")
        # Return error info for debugging if needed
        return maid, None, False

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

def process_dataset(raw_data_base, skip_existing_maids, tf_instance, maid_mapping, valid_maids, start_date=None, end_date=None, mongo_client=None, db_name='casacom', collection_name='maids'):
    """Process all data with optional date range filtering - processes in file batches to maximize RAM usage and bulk save to MongoDB"""
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

    # MongoDB setup
    if mongo_client is None:
        # Use environment variables for MongoDB connection
        mongo_host = os.getenv('MONGO_HOST', 'localhost')
        mongo_port = int(os.getenv('MONGO_PORT', '27017'))
        mongo_client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
        print(f"Connected to MongoDB at {mongo_host}:{mongo_port}")
    db = mongo_client[db_name]
    collection = db[collection_name]

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
        data = conn.execute(f"""
            SELECT maid, timestamp, latitude, longitude, flux
            FROM read_parquet(['{file_list}'])
            WHERE latitude BETWEEN -90 AND 90
            AND longitude BETWEEN -180 AND 180
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

        map_chunksize = get_env_int('MAP_CHUNKSIZE', 100)

        print(f"Processing MAIDs with {num_processes} workers...")

        # === NEW: Query existing MAIDs for current batch ===
        current_batch_maids = {maid for maid, _ in maid_data_list}
        print(f"Checking {len(current_batch_maids)} MAIDs for existing data...")

        if current_batch_maids:
            try:
                # Query existing MAIDs in batches to avoid MongoDB document size limit
                maid_list = list(current_batch_maids)
                batch_existing_maids, batch_existing_docs = batch_query_existing_maids(
                    collection, maid_list
                )
            except Exception as e:
                print(f"Warning: Could not query existing MAIDs for batch: {e}")
                batch_existing_maids = set()
                batch_existing_docs = {}
        else:
            batch_existing_maids = set()
            batch_existing_docs = {}

        # Process all maids and collect stores for bulk MongoDB save
        all_stores = []
        successful_maids = 0

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(
                executor.map(process_maid_data, maid_data_list,
                           [batch_existing_docs] * len(maid_data_list),
                           chunksize=map_chunksize),
                total=len(maid_data_list),
                desc=f"  Processing MAIDs (batch {current_batch_num}/{total_batches})"
            ))

            # Collect stores for bulk MongoDB save
            for maid, store, success in results:
                if success and store is not None:
                    all_stores.append((maid, store))
                    successful_maids += 1
        print(f"Completed processing {successful_maids:,}/{len(maid_data_list):,} MAIDs from batch {current_batch_num}/{total_batches}")

        # Bulk save to MongoDB
        if all_stores:
            print(f"Bulk saving {len(all_stores)} stores to MongoDB...")
            bulk_ops = []

            for maid, store in all_stores:
                # Convert store to dict for MongoDB with proper serialization
                from datetime import datetime as dt
                store_dict = {
                    '_id': maid,
                    'maid': store.maid,
                    'store': store._serialize_store_for_mongo(),  # Use serialized store
                    'version': '2.0',
                    'total_pings': store.total_pings
                }
                bulk_ops.append(UpdateOne(
                    {'_id': maid},
                    {'$set': store_dict},
                    upsert=True
                ))

            try:
                result = collection.bulk_write(bulk_ops)
                print(f"Bulk write completed: {result.modified_count} modified, {result.upserted_count} upserted")
            except Exception as e:
                print(f"Error during bulk write to MongoDB: {e}")
                # Fallback: save individual stores
                for maid, store in all_stores:
                    try:
                        from datetime import datetime as dt
                        store_dict = {
                            '_id': maid,
                            'maid': store.maid,
                            'store': store._serialize_store_for_mongo(),  # Use serialized store
                            'version': '2.0',
                            'total_pings': store.total_pings
                        }
                        collection.replace_one({'_id': maid}, store_dict, upsert=True)
                    except Exception as e2:
                        print(f"Error saving individual store {maid}: {e2}")

        # Collect garbage after processing each batch to free up memory
        del maid_data_list
        del results
        del all_stores
        gc.collect()  # Keep explicit gc.collect() for batch-level cleanup
    # Close DuckDB connection
    conn.close()

    print(f"\n{'='*60}")
    print(f"Completed processing all file batches")
    print(f"{'='*60}")
    


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
    # Load MAID tuple mapping for deduplication (only once in main process)
    maid_tuple_file = get_env_str('MAID_MAPPING_FILE', './maid_mapping.pkl')
    print(f"Loading MAID tuple mapping from {maid_tuple_file}...")
    maid_mapping = pickle.load(open(maid_tuple_file, 'rb'))
    valid_maids = set(maid_mapping.keys())
    print(f"Loaded {len(maid_mapping):,} MAID pairs, total {len(valid_maids):,} valid MAIDs")

    # Parse command line arguments
    start_date = None
    end_date = None
    raw_data_base = get_env_str('DATA_RAW_PATH', './data/raw_rt')  # Use env var or default to relative path

    # MongoDB configuration from environment variables
    mongo_host = get_env_str('MONGO_HOST', 'localhost')
    mongo_port = get_env_int('MONGO_PORT', 27017)
    mongo_db = get_env_str('MONGO_DB', 'casacom')
    mongo_collection = get_env_str('MONGO_COLLECTION', 'maids')

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
        elif arg == '--mongo-host' and i + 1 < len(sys.argv):
            mongo_host = sys.argv[i + 1]
            i += 1  # Skip next argument as it's the host value
        elif arg == '--mongo-port' and i + 1 < len(sys.argv):
            mongo_port = int(sys.argv[i + 1])
            i += 1  # Skip next argument as it's the port value
        elif arg == '--mongo-db' and i + 1 < len(sys.argv):
            mongo_db = sys.argv[i + 1]
            i += 1  # Skip next argument as it's the db value
        elif arg == '--mongo-collection' and i + 1 < len(sys.argv):
            mongo_collection = sys.argv[i + 1]
            i += 1  # Skip next argument as it's the collection value
        elif re.match(r'\d{4}-\d{2}-\d{2}', arg):
            # If it's a date format, treat as start date if not already set
            if start_date is None:
                start_date = parse_date_argument(arg)
            elif end_date is None:
                end_date = parse_date_argument(arg)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python process_maroc_mongo.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--raw-data PATH]")
            print("       python process_maroc_mongo.py [--mongo-host HOST] [--mongo-port PORT] [--mongo-db DB] [--mongo-collection COLLECTION]")
            print("       python process_maroc_mongo.py [start_date] [end_date]")
            print("Examples:")
            print("  python process_maroc_mongo.py")
            print("  python process_maroc_mongo.py --start-date 2025-08-05")
            print("  python process_maroc_mongo.py --start-date 2025-08-05 --end-date 2025-08-10")
            print("  python process_maroc_mongo.py 2025-08-05 2025-08-10")
            print("  python process_maroc_mongo.py --mongo-host mongodb://localhost:27017 --mongo-db casacom")
            sys.exit(1)

        i += 1

    # Create MongoDB client
    mongo_uri = f"mongodb://{mongo_host}:{mongo_port}"
    print(f"Connecting to MongoDB at {mongo_uri}, database: {mongo_db}, collection: {mongo_collection}")
    mongo_client = MongoClient(mongo_uri)

    # Note: We no longer query all existing MAIDs upfront to avoid memory usage
    # Instead, we'll query existing MAIDs per batch in process_dataset

    # Process all data with existing MAIDs set
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
    process_dataset(raw_data_base, False, None, maid_mapping, valid_maids, start_date, end_date, mongo_client, mongo_db, mongo_collection)

    # Close MongoDB connection
    mongo_client.close()

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
        os.environ['FILE_BATCH_SIZE'] = '3000'
        print("Setting FILE_BATCH_SIZE to 3000 for optimal performance")

    if not os.environ.get('MAID_QUERY_BATCH_SIZE'):
        os.environ['MAID_QUERY_BATCH_SIZE'] = '100000'
        print("Setting MAID_QUERY_BATCH_SIZE to 100000 to avoid MongoDB document size limits")

    main()
