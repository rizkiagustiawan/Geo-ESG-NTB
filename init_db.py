import os
import psycopg2

DB_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://geoesg_user:geoesg_password@db:5432/geoesg_spatial"
)

def init_db():
    """Inisialisasi tabel audit_logs di PostgreSQL + PostGIS."""
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Aktifkan ekstensi spasial
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        
        # Buat tabel enterprise dengan kolom GEOMETRY
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                site_id TEXT,
                sat_ndvi REAL,
                ground_biomass REAL,
                trust_score REAL,
                biomass REAL,
                carbon REAL,
                status TEXT,
                geom GEOMETRY(Polygon, 4326),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.close()
        conn.close()
        print("✅ Database initialized successfully.")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")

if __name__ == "__main__":
    init_db()
