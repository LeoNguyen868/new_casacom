import duckdb
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process data for a specific date")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--base_path", help="Base path")
    args = parser.parse_args()
    
    date = args.date
    base_path = args.base_path
    
    # Get environment variables for paths
    data_raw_path = os.environ.get("DATA_RAW_PATH1" if base_path == "raw" else "DATA_RAW_PATH2", f"/home/hieu/Work/new_casacom/data/{base_path}")
    output_dir = os.environ.get("OUTPUT_DIR", "/home/hieu/Work/new_casacom/data/processed")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the output path
    output_path = f"{output_dir}/{date}.parquet"
    
    # Execute the query
    duckdb.query(f"""
    copy(
    select maid, flux, count(*) as maid_flux from read_parquet('{data_raw_path}/{date}/*.parquet')
    group by flux,maid
    ) to '{output_path}' (format parquet)
    """)

if __name__ == "__main__":
    main()
