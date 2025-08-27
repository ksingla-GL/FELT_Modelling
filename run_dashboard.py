"""
Simple dashboard launcher - run this with: python run_dashboard.py
"""
import os
import subprocess
import sys

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Starting FELT Token Dashboard...")
print("Dashboard will open at http://localhost:8501")
print("Press Ctrl+C to stop")

# Run streamlit using python -m
subprocess.call([sys.executable, "-m", "streamlit", "run", "db.py", "--server.port", "8501"])