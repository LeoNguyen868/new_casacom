import pandas as pd
import pygeohash as pgh
import os
from tqdm import tqdm
from collections import defaultdict
import warnings

# Get file path from environment variable if available, otherwise use default
mrc_file = os.environ.get('MRC_FILE', './mrc_geohash.parquet')
mrc = pd.read_parquet(mrc_file)

def process_mrc(df, show_progress=True):
    """
    Optimized MRC processing with enhanced progress reporting and performance improvements.

    Args:
        df: Input dataframe with geohash and category columns
        show_progress: Whether to display tqdm progress bar
    """
    # Pre-compute filtered geohashes for better performance
    filtered_df = df[df.category.isin(['home','work','leisure'])]
    unique_geohashes = filtered_df.geohash.unique()

    print(f"Processing {len(unique_geohashes)} unique geohashes from {df.shape[0]} total records...")

    # Pre-compute geohash prefix mapping for faster lookups
    # Group MRC data by geohash prefix (first 6 characters)
    mrc_indexed = defaultdict(list)
    for _, tower in mrc.iterrows():
        prefix = tower['geohash'][:6]
        mrc_indexed[prefix].append(tower)

    # Convert to regular dict for faster access
    mrc_indexed = dict(mrc_indexed)

    all_cell_info = []
    failed_calculations = 0

    # Enhanced progress bar with more informative display
    iterator = unique_geohashes
    if show_progress:
        iterator = tqdm(unique_geohashes, desc="Processing MRC geohashes",
                       unit="geohash", total=len(unique_geohashes),
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for i in iterator:
        geohash_prefix = i[:6]
        nearby_towers = mrc_indexed.get(geohash_prefix, [])

        cell_info = {}

        if len(nearby_towers) > 0:
            for tower in nearby_towers:  # Iterate directly over list for better performance
                tower_gh = tower['geohash']
                try:
                    distance = pgh.geohash_haversine_distance(i, tower_gh)
                    cell_info[tower_gh] = {
                        'distance': distance,
                        'radio': tower.get('radio', 'unknown'),
                        'mcc': tower.get('mcc', 'unknown'),
                        'net': tower.get('net', 'unknown'),
                        'lat': tower.get('lat', 0),
                        'lon': tower.get('lon', 0),
                        'range': tower.get('range', 0)
                    }
                except Exception as e:
                    failed_calculations += 1
                    # Use warnings module for better error reporting
                    warnings.warn(f"Distance calculation failed for geohash {tower_gh}: {str(e)}",
                                category=UserWarning, stacklevel=2)
        else:
            # Handle case with no nearby towers
            cell_info[i] = {
                'distance': 0,
                'radio': 'unknown',
                'mcc': 'unknown',
                'net': 'unknown',
                'lat': 0,
                'lon': 0,
                'range': 0
            }

        all_cell_info.append({'geohash': i, 'cell_info': cell_info})

    if show_progress:
        print(f"\nProcessing complete! Failed calculations: {failed_calculations}")

    return pd.DataFrame(all_cell_info)