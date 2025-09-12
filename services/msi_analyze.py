# Install required packages before running (uncomment if needed)
# import subprocess
# subprocess.run("pip install eo-learn sentinelhub numpy matplotlib", shell=True)

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, BBox, CRS, DataCollection
from eolearn.core import EOTask, FeatureType
from eolearn.io import SentinelHubInputTask
from shapely.geometry import Polygon

# Sentinel Hub credentials (replace with your own)
config = SHConfig()
config.sh_client_id = "80f959d6-1900-438f-a09e-b751438bafa2"
config.sh_client_secret = "g8x3nTEgYPLL9njhVqKoV9pdPqpRr07I"

def analyze_farmland(coords, time_range):
    """
    coords: [min_lon, min_lat, max_lon, max_lat]
    time_range: ('YYYY-MM-DD', 'YYYY-MM-DD')
    Returns: dict of calculated indices and best timestamp
    """
    farmland_bbox = BBox(bbox=coords, crs=CRS.WGS84)

    # Download satellite data
    load_data_task = SentinelHubInputTask(
        data_collection=DataCollection.SENTINEL2_L2A,
        bands=['B02', 'B03', 'B04', 'B05', 'B08', 'B11', 'B12', 'SCL'],
        bands_feature=(FeatureType.DATA, 'BANDS'),
        resolution=10,
        maxcc=0.5,
        config=config
    )

    class SclCloudMaskTask(EOTask):
        def execute(self, eopatch):
            scl_band = eopatch.data['BANDS'][..., -1]
            cloud_classes = [3, 8, 9, 10, 11]
            cloud_mask = np.isin(scl_band, cloud_classes)
            eopatch.mask['CLM'] = np.expand_dims(cloud_mask, axis=-1).astype(bool)
            return eopatch

    class CalculateIndicesTask(EOTask):
        def execute(self, eopatch):
            np.seterr(divide='ignore', invalid='ignore')
            bands = eopatch.data['BANDS'][..., :-1]
            blue, green, red, re, nir, swir1, swir2 = bands[..., 0], bands[..., 1], bands[..., 2], bands[..., 3], bands[..., 4], bands[..., 5], bands[..., 6]
            ndvi = (nir - red) / (nir + red)
            evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
            savi = ((nir - red) / (nir + red + 0.5)) * 1.5
            ndre = (nir - re) / (nir + re)
            msi = swir1 / nir
            ndmi = (nir - swir1) / (nir + swir1)
            bare_soil_mask = ndvi < 0.25
            bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
            predicted_soc_percent = np.clip(2.25 - 1.75 * bsi, 0, 10)
            predicted_soc_percent[~bare_soil_mask] = np.nan
            predicted_om_percent = predicted_soc_percent * 1.724
            final_results = np.stack([ndvi, evi, savi, ndre, msi, ndmi, predicted_soc_percent, predicted_om_percent], axis=-1)
            eopatch.data['INDICES'] = final_results
            return eopatch

    scl_mask_task = SclCloudMaskTask()
    calculate_indices_task = CalculateIndicesTask()

    result_eopatch = None
    try:
        result_eopatch = load_data_task.execute(
            bbox=farmland_bbox,
            time_interval=time_range
        )
        if not result_eopatch.timestamp:
            return {"error": "No satellite images found for the given area and time range."}
        result_eopatch = scl_mask_task.execute(result_eopatch)
        result_eopatch = calculate_indices_task.execute(result_eopatch)
    except Exception as e:
        return {"error": str(e)}

    # Find best timestamp (least cloud coverage)
    cloud_coverage = result_eopatch.mask['CLM'].mean(axis=(1, 2, 3))
    best_timestamp_idx = np.argmin(cloud_coverage)
    best_timestamp = result_eopatch.timestamp[best_timestamp_idx]
    indices_data = result_eopatch.data['INDICES'][best_timestamp_idx]

    # Prepare output
    index_names = ['NDVI', 'EVI', 'SAVI', 'NDRE', 'MSI', 'NDMI', 'SOC', 'OM']
    output = {name: indices_data[..., i] for i, name in enumerate(index_names)}
    output['timestamp'] = str(best_timestamp)
    return output

# Example usage:
if __name__ == "__main__":
    # Example coordinates and time range
    coords = [75.836, 30.852, 75.842, 30.856]
    time_range = ('2025-07-01', '2025-07-15')
    result = analyze_farmland(coords, time_range)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Best timestamp:", result["timestamp"])
        for k, v in result.items():
            if k != "timestamp":
                print(f"{k} shape: {v.shape}")