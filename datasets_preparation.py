import glob
import re
import shutil
import stackstac
import numpy as np
import pystac_client
import planetary_computer
import rioxarray as rxr
from pathlib import Path


out_folder = Path(r"D:\Flood\Foss4gAsia2023\Assam")
input_chips_folder = "chips"  
(out_folder / 'images').mkdir(parents=True, exist_ok=True)
(out_folder / 'labels').mkdir(parents=True, exist_ok=True)

# Get all Sentinel-1 label paths
S1_label_paths = sorted(glob.glob(f"{input_chips_folder}/*/s1/*/LabelWater.tif"))
print(f"Found {len(S1_label_paths)} labeled images")

def extract_date(filename):
    """Extract date from Sentinel-1 filename"""
    match = re.search(r'S1[A,B]_IW_GRDH_1SDV_(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return None

# Initialize STAC client
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

for idx, label_path in enumerate(S1_label_paths):
    try:
        print(f"\nProcessing {idx+1}/{len(S1_label_paths)}: {label_path}")
        
        # Load and reproject label
        label = rxr.open_rasterio(label_path, masked=True)
        label_reprj = label.rio.reproject('epsg:4326')
        bbox = label_reprj.rio.bounds()

        # Get EPSG code
        epsg = label.rio.crs.to_authority()[1]

        # Extract date
        date = extract_date(label_path)
        if not date:
            print(f"Skipping {label_path} - could not extract date")
            continue
        
        print(f"Date: {date}, BBOX: {bbox}, EPSG: {epsg}")

        # Search for Sentinel-1 RTC data
        search = catalog.search(
            collections=["sentinel-1-rtc"],
            bbox=bbox,
            datetime=date
        )
        items = search.item_collection()
        
        if len(items) == 0:
            print(f"No Sentinel-1 items found for {date}")
            continue
        
        s1_item = items[0]
        print(f"Found Sentinel-1 item: {s1_item.id}")

        # Search for NASADEM elevation data
        dem_search = catalog.search(
            collections=["nasadem"],
            bbox=bbox
        )
        dem_items = dem_search.item_collection()

        if len(dem_items) == 0:
            print(f"No elevation data found for area")
            continue

        # Combine items and create stack
        all_items = [s1_item] + list(dem_items)
        ds = stackstac.stack(
            all_items,
            bounds=label.rio.bounds(),
            epsg=int(epsg),
            resolution=10
        ).mean(dim='time').to_dataset(dim='band')

        # Calculate VV/VH ratio
        ds['vv_vh'] = ds.vv / ds.vh

        # Ensure CRS is assigned before saving
        ds = ds.rio.write_crs(f"EPSG:{epsg}", inplace=True)
        label = label.rio.write_crs(f"EPSG:{epsg}", inplace=True)

        # Prepare output paths
        output_id = f"{idx:04d}"
        image_path = out_folder / 'images' / f"{output_id}.tif"
        label_path_out = out_folder / 'labels' / f"{output_id}.tif"

        # Save image
        ds[['vv', 'vh', 'vv_vh', 'elevation']].rio.to_raster(
            image_path,
            dtype=np.float32
        )

        # Save label
        label.rio.to_raster(label_path_out, dtype=np.uint8)

        print(f"Saved outputs for {output_id}")
        print(f"  Image: {image_path}")
        print(f"  Label: {label_path_out}")

    except Exception as e:
        print(f"Error processing {label_path}: {str(e)}")
        continue

print("\nProcessing complete!")
print(f"Saved {len(list((out_folder / 'images').glob('*.tif')))} images")
print(f"Saved {len(list((out_folder / 'labels').glob('*.tif')))} labels")
