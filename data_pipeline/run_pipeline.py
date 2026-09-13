import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(script_name):
    script_path = BASE_DIR / script_name
    print(f"\nRunning {script_name}...")
    subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    print("Starting complete data pipeline...")

    run_script("scraper.py")
    run_script("database.py")
    run_script("queries.py")
    run_script("pandas_analysis.py")

    print("\nComplete data pipeline executed successfully!")