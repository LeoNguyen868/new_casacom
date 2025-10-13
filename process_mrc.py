import pandas as pd
import pygeohash as pgh
import os

# Get file path from environment variable if available, otherwise use default
mrc_file = os.environ.get('MRC_FILE', './mrc_geohash.parquet')
mrc = pd.read_parquet(mrc_file)

def process_mrc(df):
    all_cell_info=[]
    print("processing...",df.shape[0])
    for i in df[df.category.isin(['home','work','leisure'])].geohash.unique():
        nearby_towers = mrc[mrc['geohash'].str.startswith(i[:6])]  # Use first 6 chars for broader search
        # Calculate distances to nearby towers
        cell_info = {}
        if len(nearby_towers) > 0:
            for _, tower in nearby_towers.iterrows():
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
                    print(e)
                    # Set default values if calculation fails
        else:
            # Fix: Don't reference tower variable when there are no nearby towers
            cell_info[i] = {  # Using geohash i instead of undefined tower_gh
                'distance': 0,
                'radio': 'unknown',
                'mcc': 'unknown',
                'net': 'unknown',
                'lat': 0,
                'lon': 0,
                'range': 0
            }
        
        all_cell_info.append({'geohash':i,'cell_info':cell_info})
    return pd.DataFrame(all_cell_info)