import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path

def tcx_to_csv(tcx_file, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(tcx_file)
    
    # Parse TCX
    tree = ET.parse(tcx_file)
    root = tree.getroot()
    
    # Garmin namespace
    ns = {'garmin': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
    
    times = []
    hrs = []
    
    # Find all trackpoints
    trackpoints = root.findall('.//garmin:Trackpoint', ns)
    
    if not trackpoints:
        print(f"Warning: No trackpoints found in {tcx_file}")
        return None
    
    for trackpoint in trackpoints:
        # Extract timestamp
        time_elem = trackpoint.find('garmin:Time', ns)
        if time_elem is None:
            continue
        time_str = time_elem.text
        
        # Extract HR
        hr_elem = trackpoint.find('.//garmin:Value', ns)
        if hr_elem is None:
            continue
        
        try:
            hr = int(hr_elem.text)
        except (ValueError, TypeError):
            continue
        
        times.append(time_str)
        hrs.append(hr)
    
    if not times:
        print(f"Warning: No HR data extracted from {tcx_file}")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({'time': times, 'hr': hrs})
    
    # Save to CSV
    base_name = os.path.basename(tcx_file).replace('.tcx', '')
    output_file = os.path.join(output_dir, f'{base_name}_hr.csv')
    
    df.to_csv(output_file, index=False)
    print(f"✓ Converted: {tcx_file} → {output_file} ({len(df)} records)")
    
    return output_file

def batch_convert_tcx(folder_path, output_dir=None):
    """
    Convert all TCX files in a folder to CSV.
    
    Args:
        folder_path: Path to folder containing .tcx files
        output_dir: Directory to save CSVs (default: same as input folder)
    """
    if output_dir is None:
        output_dir = folder_path
    
    # Find all TCX files
    tcx_files = list(Path(folder_path).glob('*.tcx'))
    
    if not tcx_files:
        print(f"No .tcx files found in {folder_path}")
        return
    
    print(f"Found {len(tcx_files)} TCX files. Converting...\n")
    
    successful = 0
    for tcx_file in sorted(tcx_files):
        try:
            result = tcx_to_csv(str(tcx_file), output_dir)
            if result:
                successful += 1
        except Exception as e:
            print(f"✗ Error processing {tcx_file}: {e}")
    
    print(f"\n✓ Successfully converted {successful}/{len(tcx_files)} files")


tcx_folder = "C:/Users/Thomas/Desktop/mlqs"

output_folder = None

batch_convert_tcx(tcx_folder, output_folder)