E-sense Car Analyzer — simple web app
Files:
- index.html, style.css : frontend (HTML + CSS + JS)
- app.py : Flask backend server that serves index.html and /api/analyze
- model_runner.py : your analysis logic (adapted). Update model paths to point to correct files on the server.
- uploads/ : folder where uploaded images are saved (created at runtime)
- requirements.txt : Python deps

How to run (local machine):
1) Create venv: python -m venv venv
2) Activate & install: pip install -r requirements.txt
3) Make sure model paths in model_runner.py point to actual model files on your machine.
4) Run backend: python app.py
5) Open http://localhost:5000 in your browser and upload an image.

Notes:
- The project expects local model files; if they are not present the analyze function will return 'model_missing' placeholders.
- For production, use Gunicorn + Nginx and HTTPS. Limit upload size as needed.