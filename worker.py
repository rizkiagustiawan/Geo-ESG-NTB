import os
import psycopg2
from celery import Celery
from pipeline_core import run_full_pipeline

# Konfigurasi Celery
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery("geoesg_tasks", broker=broker_url, backend=result_backend)

# Konfigurasi PostgreSQL
DB_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://geoesg_user:geoesg_password@db:5432/geoesg_spatial"
)

def log_audit_worker(site_id, site_raw, site_esg, ground_truth_biomass):
    """Insert audit log ke PostgreSQL dari worker."""
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO audit_logs
               (site_id, sat_ndvi, ground_biomass, trust_score, biomass, carbon, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                site_id,
                site_raw.get("satellite_ndvi_90"),
                ground_truth_biomass,
                site_esg.get("final_trust_score"),
                site_raw.get("estimated_biomass"),
                site_raw.get("estimated_carbon"),
                site_esg.get("data_integrity_flag"),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Worker logging failed: {e}")

@celery_app.task(bind=True)
def run_pipeline_task(self, user_inputs: list):
    """
    Menjalankan pipeline secara asinkron di belakang layar.
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Menyiapkan Input...'})
        
        raw_data, esg_metrics = run_full_pipeline(user_inputs)

        # Log hasil ke database (per site)
        for inp in user_inputs:
            sid = inp["site_id"]
            gt = inp.get("ground_truth_10", 150.0)
            
            site_raw = next((r for r in raw_data if r["site_id"] == sid), None)
            site_esg = next((m for m in esg_metrics if m["site_id"] == sid), None)
            
            if site_raw and site_esg:
                log_audit_worker(sid, site_raw, site_esg, gt)

        return {"raw_data": raw_data, "esg_metrics": esg_metrics}

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
