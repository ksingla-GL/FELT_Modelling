@echo off
echo Starting FELT Token Dashboard...
echo Dashboard will open at http://localhost:8501
echo Close this window to stop the dashboard
echo ================================
python -m streamlit run db.py --server.port 8501
pause