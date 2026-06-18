"""
GeoESG — GEDI L4A Ground-Truth Fetcher
======================================
Ekstraksi data Biomassa (AGB) dari satelit LiDAR NASA GEDI di ISS.
Data ini digunakan sebagai "Virtual Ground-Truth" untuk melatih model XGBoost.
"""

import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH = os.path.join(BASE_DIR, "credentials", "gee-key.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "shared_data", "gedi_groundtruth.csv")

def init_gee():
    try:
        import ee
        with open(KEY_PATH, "r") as f:
            cred_dict = json.load(f)
        credentials = ee.ServiceAccountCredentials(cred_dict["client_email"], key_file=KEY_PATH)
        ee.Initialize(credentials=credentials)
        print("✅ Koneksi GEE Berhasil.")
        return ee
    except Exception as e:
        print(f"⚠️ GEE Auth failed: {e}\nPastikan file credentials/gee-key.json tersedia.")
        return None

def fetch_gedi_data():
    ee = init_gee()
    if not ee: return

    print("🛰️ Mengunduh GEDI L4A + Sentinel/ALOS data dari Google Earth Engine...")
    # Bounding Box NTB
    roi = ee.Geometry.Rectangle([115.8, -9.1, 119.1, -8.1])

    # 1. Target (y) = GEDI AGB (Biomassa Real)
    gedi = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY") \
        .filterBounds(roi).filterDate('2021-01-01', '2021-12-31') \
        .select('agbd') \
        .median().rename('agb_true')

    # 2. Fitur Optik (Sentinel-2 NDVI)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate('2021-01-01', '2021-12-31').median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi_obs')

    # 3. Fitur SAR C-Band (Sentinel-1 VH, VV)
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi) \
        .filterDate('2021-01-01', '2021-12-31') \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .median().select(['VH', 'VV'], ['c_vh_obs', 'c_vv_obs'])

    # 4. Fitur SAR L-Band (ALOS PALSAR-2 HH, HV)
    alos = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH").filterBounds(roi).filterDate('2021-01-01', '2021-12-31').median()
    hh_db = alos.select('HH').pow(2).log10().multiply(10).subtract(83.0).rename('l_hh_obs')
    hv_db = alos.select('HV').pow(2).log10().multiply(10).subtract(83.0).rename('l_hv_obs')

    # Gabung semua band
    stacked = gedi.addBands([ndvi, s1, hh_db, hv_db])

    print("⏳ Sampling titik (25m footprint)... Mohon tunggu.")
    try:
        # NOTE: Untuk production/area besar (>10,000 titik), gunakan Export ke Google Drive:
        # task = ee.batch.Export.table.toDrive(
        #     collection=stacked.sample(region=roi, scale=25, numPixels=10000, dropNulls=True),
        #     description='GEDI_Training_Data',
        #     fileFormat='CSV'
        # )
        # task.start()
        # print("Task diekspor ke Google Drive. Cek status di code.earthengine.google.com/tasks")

        # Untuk testing interaktif (bisa memory error jika ROI/tanggal terlalu besar):
        samples = stacked.sample(
            region=roi,
            scale=250, # Kurangi resolusi spasial untuk testing lokal
            numPixels=500,
            geometries=False,
            dropNulls=True
        ).getInfo()

        features = [f['properties'] for f in samples['features']]
        df = pd.DataFrame(features)

        if not df.empty:
            # Reorder columns
            cols = ['ndvi_obs', 'c_vh_obs', 'c_vv_obs', 'l_hh_obs', 'l_hv_obs', 'agb_true']
            df = df[cols]
            
            os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Selesai! {len(df)} titik ground-truth tersimpan di {OUTPUT_CSV}")
            print("🚀 Anda sekarang bisa menjalankan: python train_xgboost_model.py")
        else:
            print("⚠️ Data kosong. ROI mungkin tertutup awan (optical) atau sensor belum cover.")
    except Exception as e:
        print(f"❌ Error saat ekstraksi: {e}")

if __name__ == "__main__":
    fetch_gedi_data()
