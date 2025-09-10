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
        "uvicorn backend.api.server:app --host 0.0.0.0 --port 8000",
        "API server"
    )
    
    # Wait a moment for API server to start
    time.sleep(70)
    
    # Start web server
    web_process = run_command(
        "uvicorn frontend.server:app --host 0.0.0.0 --port 8001",
        "web server"
    )
    
    try:
        # Keep both processes running
        api_process.wait()
        web_process.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        api_process.terminate()
        web_process.terminate()
        sys.exit(0)