# Terminal 1 — backend (Windows cmd / PowerShell)
cd backend
python -m uvicorn main:app --port 8000

# Terminal 2 — frontend
cd frontend
python -m http.server 5500
# open http://localhost:5500/sentinel_dashboard.html