from flask import Flask, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os, traceback

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXT = {'png','jpg','jpeg'}
MAX_CONTENT = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Import your analysis function from model_runner.py
try:
    from model_runner import analyze_image
    print('Imported analyze_image from model_runner')
except Exception as e:
    print('Failed to import analyze_image:', e)
    analyze_image = None

def allowed(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory('.', 'style.css')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        if 'image' not in request.files:
            return jsonify({'error':'no image part'}), 400
        f = request.files['image']
        if f.filename == '':
            return jsonify({'error':'no selected file'}), 400
        if not allowed(f.filename):
            return jsonify({'error':'file type not allowed'}), 400
        filename = secure_filename(f.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(path)
        if analyze_image is None:
            return jsonify({'error':'analysis function not available on server'}), 500
        try:
            results = analyze_image(path)
            # Ensure numeric types are standard Python types
            return jsonify(results)
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error':'failed to run analysis', 'detail': str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'server error', 'detail': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
