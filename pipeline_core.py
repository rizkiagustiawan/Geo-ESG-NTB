import os
import json
import shutil
import subprocess
import uuid
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DATA = os.path.join(BASE_DIR, "shared_data")

def run_full_pipeline(user_inputs: list, request_id: str = None):
    """
    Core pipeline logic shared between API Server and Celery Worker.
    Uses isolated temporary directories to prevent race conditions.
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]
    
    # Create isolated work directory
    work_dir = tempfile.mkdtemp(prefix=f"geoesg_run_{request_id}_")
    
    try:
        # Paths
        input_path = os.path.join(work_dir, "user_input.json")
        raw_data_path = os.path.join(work_dir, "raw_data.json")
        esg_metrics_path = os.path.join(work_dir, "esg_metrics.json")
        geojson_path = os.path.join(work_dir, "batas_ntb.geojson")

        # 1. Prepare Input
        with open(input_path, "w") as f:
            json.dump(user_inputs, f, indent=4)
        
        # Copy GeoJSON if available for extractor
        std_geojson = os.path.join(SHARED_DATA, "batas_ntb.geojson")
        if os.path.exists(std_geojson):
            shutil.copy2(std_geojson, geojson_path)

        # 2. Step 1: Python Extractor
        env = os.environ.copy()
        env["GEOESG_INPUT_PATH"] = input_path
        env["GEOESG_OUTPUT_PATH"] = raw_data_path
        env["GEOESG_GEOJSON_PATH"] = geojson_path
        
        extractor_script = os.path.join(BASE_DIR, "python-gee-ai", "extractor.py")
        venv_python = os.path.join(BASE_DIR, "venv", "bin", "python3")
        python_exe = venv_python if os.path.exists(venv_python) else "python3"
        
        res_py = subprocess.run(
            [python_exe, extractor_script],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=BASE_DIR,
            env=env
        )
        if res_py.returncode != 0:
            raise RuntimeError(f"Python Extractor Error: {res_py.stderr}")

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Extractor failed to produce {raw_data_path}")

        # 3. Step 2: Rust ESG Engine
        rust_binary = os.path.join(BASE_DIR, "rust-esg-engine", "target", "release", "rust-esg-engine")
        
        if os.path.exists(rust_binary):
            # Pass absolute paths to avoid relative path confusion
            res_rs = subprocess.run(
                [rust_binary, os.path.abspath(raw_data_path), os.path.abspath(esg_metrics_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.join(BASE_DIR, "rust-esg-engine")
            )
        else:
            # Fallback to cargo run
            res_rs = subprocess.run(
                ["cargo", "run", "--release", "-q", "--", os.path.abspath(raw_data_path), os.path.abspath(esg_metrics_path)],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=os.path.join(BASE_DIR, "rust-esg-engine")
            )
            
        if res_rs.returncode != 0:
            raise RuntimeError(f"Rust Engine Error: {res_rs.stderr}")

        if not os.path.exists(esg_metrics_path):
            print(f"DEBUG Rust STDOUT: {res_rs.stdout}")
            print(f"DEBUG Rust STDERR: {res_rs.stderr}")
            raise FileNotFoundError(f"Rust Engine failed to produce {esg_metrics_path}")

        # 4. Load results
        with open(raw_data_path, "r") as f:
            raw_data = json.load(f)
        with open(esg_metrics_path, "r") as f:
            esg_metrics = json.load(f)

        # 5. Sync to shared_data for backward compatibility/global view (Atomic Replace)
        tmp_raw = os.path.join(SHARED_DATA, f"raw_{request_id}.json")
        tmp_esg = os.path.join(SHARED_DATA, f"esg_{request_id}.json")
        shutil.copy2(raw_data_path, tmp_raw)
        shutil.copy2(esg_metrics_path, tmp_esg)
        os.replace(tmp_raw, os.path.join(SHARED_DATA, "raw_data.json"))
        os.replace(tmp_esg, os.path.join(SHARED_DATA, "esg_metrics.json"))

        return raw_data, esg_metrics

    finally:
        # Cleanup isolated folder
        shutil.rmtree(work_dir, ignore_errors=True)
