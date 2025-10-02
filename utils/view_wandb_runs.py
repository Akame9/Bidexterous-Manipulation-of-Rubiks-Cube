"""
Utility script to view and manage Weights & Biases runs.
This script helps you easily access your wandb dashboard links.
"""

import os
import glob
from datetime import datetime


def list_wandb_runs():
    """List all wandb runs saved in logs directory."""
    log_files = glob.glob("logs/wandb_run_*.txt")
    
    if not log_files:
        print("No wandb runs found in logs directory.")
        return
    
    print("Available Weights & Biases Runs:")
    print("=" * 50)
    
    # Sort by creation time (newest first)
    log_files.sort(key=os.path.getctime, reverse=True)
    
    for i, log_file in enumerate(log_files, 1):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Extract key information
            run_name = ""
            project = ""
            url = ""
            started = ""
            
            for line in lines:
                if line.startswith("Run Name:"):
                    run_name = line.split(":", 1)[1].strip()
                elif line.startswith("Project:"):
                    project = line.split(":", 1)[1].strip()
                elif line.startswith("URL:"):
                    url = line.split(":", 1)[1].strip()
                elif line.startswith("Started:"):
                    started = line.split(":", 1)[1].strip()
            
            print(f"{i}. {run_name}")
            print(f"   Project: {project}")
            print(f"   Started: {started}")
            print(f"   URL: {url}")
            print(f"   Log file: {log_file}")
            print()
            
        except Exception as e:
            print(f"Error reading {log_file}: {e}")


def get_latest_wandb_run():
    """Get the latest wandb run URL."""
    log_files = glob.glob("logs/wandb_run_*.txt")
    
    if not log_files:
        print("No wandb runs found.")
        return None
    
    # Get the most recent file
    latest_file = max(log_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith("URL:"):
                url = line.split(":", 1)[1].strip()
                print(f"Latest wandb run: {url}")
                return url
                
    except Exception as e:
        print(f"Error reading latest run: {e}")
    
    return None


def open_wandb_run(run_number=None):
    """Open a wandb run in the browser."""
    try:
        import webbrowser
    except ImportError:
        print("webbrowser module not available")
        return
    
    log_files = glob.glob("logs/wandb_run_*.txt")
    
    if not log_files:
        print("No wandb runs found.")
        return
    
    # Sort by creation time (newest first)
    log_files.sort(key=os.path.getctime, reverse=True)
    
    if run_number is None:
        # Open latest run
        target_file = log_files[0]
        print("Opening latest wandb run...")
    else:
        if run_number < 1 or run_number > len(log_files):
            print(f"Invalid run number. Available runs: 1-{len(log_files)}")
            return
        target_file = log_files[run_number - 1]
    
    try:
        with open(target_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith("URL:"):
                url = line.split(":", 1)[1].strip()
                print(f"Opening: {url}")
                webbrowser.open(url)
                return
                
    except Exception as e:
        print(f"Error opening run: {e}")


def clean_old_logs(days=30):
    """Clean wandb log files older than specified days."""
    import time
    
    log_files = glob.glob("logs/wandb_run_*.txt")
    current_time = time.time()
    cutoff_time = current_time - (days * 24 * 60 * 60)
    
    removed_count = 0
    for log_file in log_files:
        if os.path.getctime(log_file) < cutoff_time:
            try:
                os.remove(log_file)
                removed_count += 1
                print(f"Removed old log: {log_file}")
            except Exception as e:
                print(f"Error removing {log_file}: {e}")
    
    if removed_count == 0:
        print(f"No log files older than {days} days found.")
    else:
        print(f"Removed {removed_count} old log files.")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="View and manage Weights & Biases runs")
    parser.add_argument('--list', action='store_true', help='List all wandb runs')
    parser.add_argument('--latest', action='store_true', help='Show latest wandb run URL')
    parser.add_argument('--open', type=int, nargs='?', const=0, help='Open wandb run in browser (latest if no number specified)')
    parser.add_argument('--clean', type=int, default=30, help='Clean log files older than N days (default: 30)')
    
    args = parser.parse_args()
    
    if args.list:
        list_wandb_runs()
    elif args.latest:
        get_latest_wandb_run()
    elif args.open is not None:
        if args.open == 0:
            open_wandb_run()
        else:
            open_wandb_run(args.open)
    elif args.clean:
        clean_old_logs(args.clean)
    else:
        # Default: list runs
        list_wandb_runs()


if __name__ == "__main__":
    main()
