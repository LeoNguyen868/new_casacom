import os
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import argparse
import sys

def parse_date_argument(date_str):
    """Parse date string in YYYY-MM-DD format"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Invalid date format: {date_str}. Please use YYYY-MM-DD format.")
        sys.exit(1)

def print_progress(completed_count, total_count):
    """Print progress to terminal"""
    print(f"\rProgress: {completed_count}/{total_count} dates processed", end="", flush=True)

def print_download_status(active_downloads, completed_count, total_count):
    """Print current download status"""
    print("\n" + "="*60)
    print("Active Downloads:")
    
    # Show the most recent 10 downloads
    recent_downloads = list(active_downloads.items())[-10:]
    for date, status in recent_downloads:
        status_symbol = "Done" if status == "Completed" else "Failed" if "Failed" in status else "In progress"
        print(f"  {status_symbol} {date}: {status}")
    
    print(f"\nProgress: {completed_count}/{total_count} dates processed")
    print("="*60)

def download_day(date, bucket, local_base_dir, endpoint_url, profile, lock, active_downloads):
    """Download data for a specific date"""
    source = f"{bucket}/{date}/"
    target = f"{local_base_dir}/{date}/"
    
    # Create directory if it doesn't exist
    os.makedirs(target, exist_ok=True)
    
    # Update active downloads
    with lock:
        active_downloads[date] = "In progress"
    
    # Construct and execute AWS command
    cmd = f"aws s3 --endpoint-url {endpoint_url} sync {source} {target} --profile {profile}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Update status
    with lock:
        if result.returncode == 0:
            active_downloads[date] = "Completed"
        else:
            active_downloads[date] = f"Failed: {result.stderr[:50]}..."
    
    return date, result.returncode

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download data from S3 bucket")
    parser.add_argument("--bucket", default="s3://home-work/hw/project2_rt", help="S3 bucket path")
    parser.add_argument("--local-dir", default="./data/raw_rt", help="Local directory to store downloaded data")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel downloads")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format (defaults to start date if not provided)")
    parser.add_argument("--endpoint-url", default="https://s3.gra.io.cloud.ovh.net", help="S3 endpoint URL")
    parser.add_argument("--profile", default="hw", help="AWS profile name")
    
    args = parser.parse_args()
    
    # Define parameters
    endpoint_url = args.endpoint_url
    bucket = args.bucket
    local_base_dir = args.local_dir
    profile = args.profile
    num_workers = args.workers
    
    # Parse dates
    start_date = parse_date_argument(args.start_date)
    end_date = parse_date_argument(args.end_date) if args.end_date else start_date
    
    # Generate dates
    delta = end_date - start_date
    dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
    
    # Track progress and active downloads
    completed_count = 0
    active_downloads = {}
    lock = threading.Lock()
    
    # Print initial status
    print(f"Starting download of {len(dates)} dates using {num_workers} workers...")
    print("="*60)
    
    # Execute downloads in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_day, date, bucket, local_base_dir, endpoint_url, profile, lock, active_downloads) for date in dates]
        
        # Periodically update status display
        while completed_count < len(dates):
            time.sleep(5)  # Update every 5 seconds
            
            # Count completed tasks
            completed_count = sum(1 for future in futures if future.done())
            
            if completed_count > 0:  # Only print status if some work has been done
                print_download_status(active_downloads, completed_count, len(dates))
        
        # Wait for all downloads to complete
        for future in futures:
            future.result()
    
    # Print final summary
    print("\n" + "="*60)
    successful = [date for date, status in active_downloads.items() if status == "Completed"]
    failed = [date for date, status in active_downloads.items() if "Failed" in status]
    
    print("DOWNLOAD SUMMARY:")
    print(f"Successfully downloaded: {len(successful)} days")
    print(f"Failed downloads: {len(failed)} days")
    
    if failed:
        print(f"  Failed dates: {', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}")
    
    print("="*60)
    print("Download process completed!")

if __name__ == "__main__":
    main()