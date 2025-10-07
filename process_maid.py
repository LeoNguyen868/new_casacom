import glob
import pandas as pd
import numpy as np
from process_poi import process_maid_data
from process_mrc import process_mrc
from process_country import get_country_from_coordinates
import gc
import multiprocessing as mp
from tqdm import tqdm
import os

def process_single_file(file):
    res, maid_id = process_maid_data(file)
    if res.empty or maid_id is None:
        return None
    res['maid'] = maid_id
    return res

def process_batch(file_batch, batch_id):
    batch_results = []
    
    # Create a pool of workers
    num_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} cores for parallel processing in batch {batch_id}")
    
    # Process files in parallel
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_file, file_batch),
            total=len(file_batch),
            desc=f"Processing files in batch {batch_id}"
        ))
        
        # Filter out None results
        batch_results = [res for res in results if res is not None]
    
    if batch_results:
        df = pd.concat(batch_results, ignore_index=True)
        
        # Process MRC data for this batch
        print(f"Processing MRC data for batch {batch_id}...")
        mrc = process_mrc(df)
        
        # Merge datasets
        print(f"Merging datasets for batch {batch_id}...")
        merged = df.merge(mrc, on='geohash', how='left')
        del df, mrc
        gc.collect()
        
        # Get country information
        print(f"Getting country information for batch {batch_id}...")
        tqdm.pandas(desc=f"Getting countries for batch {batch_id}")
        merged['country'] = merged.progress_apply(lambda x: get_country_from_coordinates(x['lat'], x['lon']), axis=1)
        
        # Process path data
        merged['coordinate'] = merged.apply(lambda x: (x['lat'], x['lon']), axis=1)
        print(f"Aggregating path data for batch {batch_id}...")
        path_pdf = merged[merged.category=='path'].groupby('maid').agg({
            'coordinate': lambda x: list(x),
            'unique_days': 'sum',
            'pings': 'sum',
            'est_duration': 'sum',
            'confidence': 'mean',
            'category': 'first',
            'country': lambda x: set(x),
            'poi_info': list
        })
        # Clean up merged dataframe
        merged.drop('coordinate', axis=1, inplace=True)
        # Calculate total flux per maid
        maid_flux=merged.groupby('maid').agg({
                'fluxB':'sum',
                'fluxC':'sum',
                'fluxD':'sum',
                'fluxE':'sum',
                'fluxF':'sum'
                }).reset_index()
        
        # Count unique countries for each maid
        country_count = merged.groupby('maid').agg({
            'country': lambda x: set(x.unique())
            }).reset_index()
        
        # Create dictionary of country:pings counts for each maid
        pings_each_country = merged.groupby('maid').apply(
            lambda x: {country: sum(x[x['country'] == country]['pings']) for country in x['country'].unique()}
        ).reset_index(name='country_pings')
        
        # Get first and last seen timestamps for each maid
        maid_first_last=merged.groupby('maid').agg({
            "first_seen":'min',
            "last_seen":'max'
        }).reset_index()
        
        # Select columns to keep
        merged = merged[['maid', 'geohash', 'category', 'confidence', 'country', 'lat', 'lon', 'first_seen', 'last_seen', 'est_duration',
               'pings', 'unique_days', 'evidence_score','fluxB', 'fluxC', 'fluxD', 'fluxE', 'fluxF',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
               'saturday', 'sunday', 'month_1', 'month_2', 'month_3', 'month_4',
               'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10',
               'month_11', 'month_12', 'hour_0', 'hour_1', 'hour_2', 'hour_3', 
               'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 
               'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 
               'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 
               'hour_22', 'hour_23', 'poi_score', 'poi_info',
               'home', 'work', 'leisure', 'path','maid_total_pings','cell_info']]
        
        # Merge all dataframes together
        merged = pd.merge(merged, maid_flux, on='maid', how='left', suffixes=('', '_maid'))
        merged = pd.merge(merged, country_count, on='maid', how='left', suffixes=('', '_maid'))
        merged = pd.merge(merged, pings_each_country, on='maid', how='left', suffixes=('', '_maid'))
        merged = pd.merge(merged, maid_first_last, on='maid', how='left', suffixes=('', '_maid'))
        
        # Rename columns with _maid suffix to maid_ prefix
        rename_dict = {
            'fluxB_maid': 'maid_fluxB',
            'fluxC_maid': 'maid_fluxC',
            'fluxD_maid': 'maid_fluxD',
            'fluxE_maid': 'maid_fluxE',
            'fluxF_maid': 'maid_fluxF',
            'country_maid': 'maid_countries',
            'first_seen_maid': 'maid_first_seen',
            'last_seen_maid': 'maid_last_seen',
            "country_pings": "maid_countries_pings"
        }
        
        merged = merged.rename(columns=rename_dict)
        
        # Filter by category
        print(f"Filtering data by category for batch {batch_id}...")
        work_pdf = merged[merged.category=='work']
        home_pdf = merged[merged.category=='home']
        leisure_pdf = merged[merged.category=='leisure']
        path_pdf.reset_index(inplace=True)
        
        # Create result directory if it doesn't exist
        result_dir = os.environ.get('RESULT_DIR', './result')
        os.makedirs(result_dir, exist_ok=True)
        
        # Save batch results
        print(f"Saving results for batch {batch_id}...")
        path_pdf.to_csv(f'{result_dir}/path_batch_{batch_id}.csv', index=False)
        merged.to_csv(f'{result_dir}/path_new_batch_{batch_id}.csv', index=False)
        work_pdf.to_csv(f'{result_dir}/work_batch_{batch_id}.csv', index=False)
        home_pdf.to_csv(f'{result_dir}/home_batch_{batch_id}.csv', index=False)
        leisure_pdf.to_csv(f'{result_dir}/leisure_batch_{batch_id}.csv', index=False)
        
        print(f"Batch {batch_id} processing complete!")
        
        # Free memory
        del merged, path_pdf, work_pdf, home_pdf, leisure_pdf
        gc.collect()
        
        return True
    return False

def main():
    # Get output directory from environment variables or use default
    output_dir = os.environ.get('OUTPUT_DIR', './data/processed')
    
    # Get all files
    files = glob.glob(f'{output_dir}/*.pkl')
    batch_size = 100000
    file_batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]

    # Calculate total number of batches
    total_batches = len(file_batches)
    print(f"Total number of batches to process: {total_batches}")

    # Process each batch with its ID
    for batch_id, file_batch in enumerate(file_batches):
        print(f"Processing batch {batch_id} with {len(file_batch)} files")
        process_batch(file_batch, batch_id)

    print("All batches processing complete!")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    main()