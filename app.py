from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)  # Enables CORS for communication with your HTML

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_excel(file, engine='openpyxl')
        df.fillna('', inplace=True)  # Avoid NaNs in response
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
