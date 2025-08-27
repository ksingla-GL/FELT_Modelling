#!/usr/bin/env python3
"""
FELT Token Dashboard Launcher
Launches the Streamlit dashboard for the FELT Token Financial Model
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    try:
        # Get the directory where this script is located
        if hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            script_dir = os.getcwd()
        else:
            # Running as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to db.py
        db_path = os.path.join(script_dir, 'db.py')
        
        # Check if db.py exists
        if not os.path.exists(db_path):
            print(f"Error: db.py not found at {db_path}")
            print(f"Looking in directory: {script_dir}")
            input("Press Enter to exit...")
            return
        
        print("Starting FELT Token Dashboard...")
        print("Opening browser at http://localhost:8501")
        print("Close this window to stop the dashboard")
        print("-" * 50)
        
        # Change to the correct directory
        os.chdir(script_dir)
        
        # Launch Streamlit - let it handle browser opening
        cmd = [
            sys.executable, "-m", "streamlit", "run", "db.py",
            "--server.port", "8501"
        ]
        
        # Run Streamlit and let it handle everything
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()