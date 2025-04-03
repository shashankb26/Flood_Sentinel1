import dask
import dask.config
import glob
import os
import re
import shutil
import stackstac
import numpy as np
import pystac_client
import planetary_computer
import rioxarray as rxr
from pathlib import Path
import sys

dask.config.set(scheduler='threads')
if 'distributed' in sys.modules:
    print("WARNING: distributed module is loaded unexpectedly")


base_path = "D:/Flood/Shashank/c2smsfloods/chips"
S1_label_paths = sorted(glob.glob(os.path.join(base_path, "*/s1/*/LabelWater.tif")))
S1_img_paths = [[path.replace("LabelWater.tif", "VV.tif"), path.replace("LabelWater.tif", "VH.tif")] for path in S1_label_paths]

Path('images').mkdir(exist_ok=True)
Path('labels').mkdir(exist_ok=True)

# Function to extract date from filename
def extract_date(filename):
    match = re.search(r'S1[A,B]_IW_GRDH_1SDV_(\d{8})', filename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return None

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

for idx, iw in enumerate(S1_label_paths):
    try:
        my_arr = rxr.open_rasterio(iw)
        reprj_iw = my_arr.rio.reproject('epsg:4326')
        epsg = my_arr.rio.crs
        bbox = reprj_iw.rio.bounds()
        bounds = my_arr.rio.bounds()
        date = extract_date(iw)

        if not date:
            print(f"Skipping {idx} due to invalid date extraction.")
            continue

        # Search for Sentinel-1 RTC data
        search = catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=date)
        items = search.item_collection()

        if len(items) == 0:
            print(f"Skipping {idx}, no Sentinel-1 RTC data found.")
            continue

        print(f"Processing {idx}, found {len(items)} Sentinel-1 items.")
        item = items[0]

        # Search for NASADEM data
        dem_search = catalog.search(collections=["nasadem"], bbox=bbox)
        dem_items = dem_search.item_collection()

        if len(dem_items) == 0:
            print(f"Skipping {idx}, no NASADEM data found.")
            continue
        
        items.items.append(dem_items[0])

        # Stack and process the data
        ds = stackstac.stack(items, bounds=bounds, epsg=int(epsg.to_authority()[1]), resolution=10).mean(dim='time')
        ds = ds.to_dataset(dim='band')
        ds['vv_vh'] = ds.vv / ds.vh

        # Select required bands and compute using Dask
        ds_subset = ds[['vv', 'vh', 'vv_vh', 'elevation']]
        ds_computed = dask.compute(ds_subset, scheduler='threads')[0]  

        output_image_path = f'images/{idx}.tif'
        ds_computed.rio.write_crs(epsg).rio.to_raster(output_image_path, dtype=np.float32)
        shutil.copyfile(iw, f"labels/{idx}.tif")

        print(f"Saved processed image: {output_image_path}")

    except Exception as e:
        print(f"Error processing {idx}: {e}")
