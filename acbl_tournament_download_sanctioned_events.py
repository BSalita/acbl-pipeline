"""
ACBL Tournament Events Downloader

Downloads tournament event data from the ACBL API and saves as JSON files.

USAGE:
    python acbl_tournament_download_to_json.py                    # Download all events (default)
    python acbl_tournament_download_to_json.py --start-date 2024-01-01
    python acbl_tournament_download_to_json.py --limit 100

REQUIREMENTS:
    - ACBL API key in environment variable ACBL_API_KEY
    - Or in .env file in the current directory

API NOTES:
    - Rate limited to ~60 requests per minute
    - ACBL servers may be offline Saturdays 6pm CST for 24 hours
    - Get API key at https://api.acbl.org
"""

import pathlib
import sys
import os
import time
import json
import urllib.parse
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from acbl_tournament_download_sessions_using_sanctioned_events import (
    AcblApiAuthError,
    expired_api_key_message,
)

# Load environment variables
load_dotenv()

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


def download_events(
    api_key: str,
    output_dir: pathlib.Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    sleep_seconds: float = 1.0,
    page_size: int = 50
) -> int:
    """
    Download tournament events from ACBL API.
    
    Args:
        api_key: ACBL API bearer token
        output_dir: Directory to save JSON files
        start_date: Earliest date to fetch (YYYY-MM-DD), default: 2013-01-01
        end_date: Latest date to fetch (YYYY-MM-DD), default: today
        limit: Maximum number of events to download (None = unlimited)
        sleep_seconds: Delay between API calls (default: 1.0)
        page_size: Number of events per API page (default: 50)
        
    Returns:
        Number of events written
    """
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    timeout = 30
    
    # Ensure output directory exists
    events_dir = output_dir.joinpath('tournaments/events/')
    events_dir.mkdir(parents=True, exist_ok=True)
    
    existing_count = len(list(events_dir.glob('*.sanction.json'))) + len(list(events_dir.glob('*.sanction.sql')))
    print(f"Output directory: {events_dir}")
    print(f"Existing files: {existing_count}")
    
    # Date range
    end_dt = datetime.now() if not end_date else datetime.strptime(end_date, '%Y-%m-%d')
    initial_start_dt = datetime(2013, 1, 1) if not start_date else datetime.strptime(start_date, '%Y-%m-%d')
    
    print(f"Date range: {initial_start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    
    api_base = 'https://api.acbl.org/v1/tournament/event_query'
    
    get_count = 0
    write_count = 0
    skip_count = 0
    error_count = 0
    start_time = time.time()
    current_date = end_dt
    
    try:
        while current_date >= initial_start_dt:
            # Process one month at a time (going backwards)
            month_start = current_date.replace(day=1)
            month_start_str = month_start.strftime('%Y-%m-%d')
            month_end_str = current_date.strftime('%Y-%m-%d')
            
            print(f"\n--- Processing {month_start.strftime('%Y-%m')} ---")
            
            # Build initial query
            query = {
                'page': 1,
                'page_size': page_size,
                'start_date': month_start_str,
                'end_date': month_end_str
            }
            url = f"{api_base}?{urllib.parse.urlencode(query)}"
            
            while url:
                get_count += 1
                rate = round((time.time() - start_time) / get_count, 2) if get_count > 0 else 0
                print(f"[{get_count}] rate:{rate}s/req url:{url[:80]}...")
                
                # Rate limiting
                if get_count > 1:
                    time.sleep(sleep_seconds)
                
                try:
                    response = requests.get(url, headers=headers, timeout=timeout)
                    
                    # Handle error codes
                    if response.status_code in (401, 403):
                        raise AcblApiAuthError(
                            expired_api_key_message(api_key)
                            or (
                                f"ERROR: HTTP {response.status_code} from ACBL API; "
                                "ACBL_API_KEY is invalid or expired."
                            )
                        )

                    if response.status_code in [400, 500, 504]:
                        print(f"  ERROR: HTTP {response.status_code} - skipping month")
                        error_count += 1
                        break
                    
                    if response.status_code == 429:
                        print(f"  RATE LIMITED: HTTP 429 - waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    
                    if response.status_code != 200:
                        print(f"  ERROR: HTTP {response.status_code}")
                        error_count += 1
                        break
                    
                    json_response = response.json()
                    url = json_response.get('next_page_url')
                    
                    # Process events in response
                    for data in json_response.get('data', []):
                        sanction_id = data.get('id')
                        if not sanction_id:
                            continue
                        
                        # Check if already exists (either .sql or .json)
                        file_json = events_dir / f"{sanction_id}.sanction.json"
                        file_sql = events_dir / f"{sanction_id}.sanction.sql"
                        
                        if file_sql.exists() or file_json.exists():
                            skip_count += 1
                            continue
                        
                        # Write new file
                        write_count += 1
                        print(f"  [{write_count}] Writing: {file_json.name}")
                        
                        with open(file_json, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        # Check limit
                        if limit and write_count >= limit:
                            print(f"\nLimit reached ({limit} events)")
                            return write_count
                    
                except requests.exceptions.Timeout:
                    print(f"  TIMEOUT - skipping")
                    error_count += 1
                    break
                except requests.exceptions.RequestException as e:
                    print(f"  ERROR: {e}")
                    error_count += 1
                    break
            
            # Move to previous month
            current_date = month_start - timedelta(days=1)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    return write_count


def main():
    """Main entry point"""
    import argparse
    
    # Check for API key
    api_key = os.getenv('ACBL_API_KEY')
    
    parser = argparse.ArgumentParser(
        description='Download ACBL tournament events to JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python acbl_tournament_download_to_json.py
  python acbl_tournament_download_to_json.py --start-date 2024-01-01
  python acbl_tournament_download_to_json.py --limit 100 --sleep 2

Environment:
  ACBL_API_KEY    Required. Get from https://api.acbl.org
"""
    )
    
    parser.add_argument(
        '--start-date',
        default=None,
        help='Earliest date to fetch (YYYY-MM-DD, default: 2013-01-01)'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='Latest date to fetch (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of NEW events to download (default: unlimited)'
    )
    parser.add_argument(
        '--sleep',
        type=float,
        default=1.0,
        help='Seconds between API requests (default: 1.0)'
    )
    parser.add_argument(
        '--page-size',
        type=int,
        default=50,
        help='Events per API page (default: 50, max ~200)'
    )
    parser.add_argument(
        '--output-dir',
        default='.',
        help=f'Output subdirectory relative to {acblPath} (default: .)'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='ACBL API key (default: from ACBL_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Get API key
    if args.api_key:
        api_key = args.api_key
    
    if not api_key:
        print("ERROR: ACBL_API_KEY environment variable not set")
        print("Set it with: export ACBL_API_KEY=your_key_here")
        print("Or use --api-key argument")
        print("Get an API key at: https://api.acbl.org")
        return 1
    expiry_msg = expired_api_key_message(api_key)
    if expiry_msg:
        print(expiry_msg)
        return 1
    
    # Resolve output directory
    output_dir = acblPath.joinpath(args.output_dir)
    
    print("=" * 70)
    print("ACBL Tournament Events Downloader")
    print("=" * 70)
    print()
    print(f"Output: {output_dir}")
    print(f"Date range: {args.start_date or '2013-01-01'} to {args.end_date or 'today'}")
    print(f"Limit: {args.limit or 'unlimited'}")
    print(f"Sleep: {args.sleep}s between requests")
    print()
    
    from mlBridge import print_started, print_ended
    program_start = print_started()
    print()

    try:
        write_count = download_events(
            api_key=api_key,
            output_dir=output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            sleep_seconds=args.sleep,
            page_size=args.page_size
        )
    except AcblApiAuthError as exc:
        print(exc)
        return 1

    print()
    print("=" * 70)
    print(f"COMPLETE: {write_count} events written")
    print_ended(program_start)
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
