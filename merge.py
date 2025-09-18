import os
import glob
import pandas as pd
from tqdm import tqdm

def merge_batch_files():
    """
    Merge all batch result files into 4 final files (path, work, home, leisure)
    and delete the batch files after successful merging.
    """
    print("Starting to merge batch files...")
    
    # Define categories to process
    categories = ['path', 'work', 'home', 'leisure']
    
    # Get result directory from environment variables or use default
    result_dir = os.environ.get('RESULT_DIR', './result')
    os.makedirs(result_dir, exist_ok=True)
    
    for category in categories:
        print(f"Processing {category} files...")
        
        # Get all batch files for this category
        batch_files = glob.glob(f'{result_dir}/{category}_batch_*.csv')
        
        if not batch_files:
            print(f"No {category} batch files found to merge.")
            continue
            
        print(f"Found {len(batch_files)} {category} batch files to merge.")
        
        # Read and concatenate all batch files
        dfs = []
        for file in tqdm(batch_files, desc=f"Reading {category} batch files"):
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not dfs:
            print(f"No valid data found for {category}.")
            continue
            
        # Merge all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Save merged file
        output_file = f'{result_dir}/{category}_final.csv'
        print(f"Saving merged {category} data to {output_file}...")
        merged_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(merged_df)} records to {output_file}")
        
        # Delete batch files after successful merge
        print(f"Deleting {category} batch files...")
        for file in batch_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    
    print("Merge process complete!")

if __name__ == "__main__":
    merge_batch_files()