#forked from https://github.com/traitlab/lefolab-utils/blob/main/DJI/GPStime2UTCtime.py all credit to them
import os
from datetime import datetime, timedelta

def gps2utc(gps_week, gps_ms):
    # GPS Epoch: January 6, 1980
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Convert milliseconds to seconds
    gps_seconds = gps_ms / 1000.0

    # Compute GPS time
    gps_time = gps_epoch + timedelta(weeks=gps_week, seconds=gps_seconds)

    # Convert to UTC by subtracting leap seconds
    leap_seconds = 18  # As of 2024; check if updated in future
    utc_time = gps_time - timedelta(seconds=leap_seconds)

    return utc_time

def process_dat_file(input_path):
    """
    Process a DAT file, convert GPS time to UTC time, and save to a text file.
    The output file is named using the first valid UTC timestamp.
    """
    first_valid_utc = None
    processed_lines = []
    
    # First pass - read and process all lines, find first valid timestamp
    with open(input_path, 'rb') as infile:
       for i, line in enumerate(infile):
            try:
                # Attempt decoding; ignore undecodable bytes
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if decoded_line:
                    print(f"Line {i}: {decoded_line}")
                else:
                    print(f"Line {i}: [empty after decoding]")
            except Exception as e:
                print(f"Line {i}: Could not decode - {e}")
            if line.startswith('bestpos:'):
                parts = line.strip().split(',')
                
                # Skip lines with no GPS data (bestpos:0)
                if parts[0] == 'bestpos:0' or len(parts) < 2:
                    continue
                
                # Extract GPS week and milliseconds
                try:
                    gps_week = int(parts[0].split(':')[1])
                    gps_ms_str = parts[1].split('ms')[0]
                    gps_ms = int(gps_ms_str)
                    
                    # Convert GPS time to UTC
                    utc_time = gps2utc(gps_week, gps_ms)
                    
                    # Format the UTC time for the line
                    utc_str = utc_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    # Store the first valid UTC time for the filename
                    if first_valid_utc is None:
                        first_valid_utc = utc_time
                    
                    # Rebuild the line with UTC time
                    new_line = f"{line.strip()} [UTC: {utc_str}]\n"
                    processed_lines.append(new_line)
                except (ValueError, IndexError) as e:
                    print(f"Error processing line: {line.strip()}")
                    print(f"Error details: {e}")
    
    # If no valid GPS data was found, return early
    if first_valid_utc is None:
        print(f"No valid GPS data found in {input_path}")
        return None
    
    # Create output filename using the first valid UTC time and number from the original filename
    filename_datetime = first_valid_utc.strftime("%Y%m%d_%H%M%S")
    file_number = os.path.basename(input_path)[3:6]
    output_filename = f"{filename_datetime}_UTC_{file_number}.txt"
    output_path = os.path.join(os.path.dirname(input_path), output_filename)
    
    # Write processed lines to the output file
    with open(output_path, 'w') as outfile:
        outfile.writelines(processed_lines)
    
    return output_path

def process_all_dat_files(input_dir):
    """
    Process all DAT files in the input directory and save results to the same directory.
    """
    # Get all .DAT files in the input directory
    dat_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.dat')]
    
    if not dat_files:
        print(f"No .DAT files found in {input_dir}")
        return
    
    print(f"Found {len(dat_files)} .DAT files to process")
    
    for dat_file in dat_files:
        input_path = os.path.join(input_dir, dat_file)
        
        print(f"Processing {dat_file}...")
        output_path = process_dat_file(input_path)
        
        if output_path:
            print(f"Saved results to {output_path}")
        else:
            print(f"Could not process {dat_file} - no valid GPS data")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = input("Enter the path to the directory containing .DAT files: ")
    
    process_all_dat_files(input_directory)
    print("All files processed successfully!")