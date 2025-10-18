# run.py
import subprocess
import sys
import time



def run_command(command, description):
    print(f"Starting {description}...")
    process = subprocess.Popen(command, shell=True)
    return process

if __name__ == "__main__":
    # Start API server
    api_process = run_command(
        "uvicorn backend.api.statistics_api:app --host 0.0.0.0 --port 8002",
        "API server"
    )

    
    try:
        # Keep both processes running
        api_process.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        api_process.terminate()
        sys.exit(0)