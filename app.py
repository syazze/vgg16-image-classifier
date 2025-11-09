from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# updated path for .keras
print("Checking models...")

import gdown
os.makedirs("models", exist_ok=True)

model_right_path = "models/vgg16_fine_tuned_right.keras"
model_left_path = "models/vgg16_fine_tuned_left.keras"

right_url = "https://drive.google.com/uc?id=17AmUDIx_VctABo9WutOphRHkSNGw3Dxr"
left_url = "https://drive.google.com/uc?id=1jxChWv1tRdecaPLCx45mO0BxbIOyOkft"

if not os.path.exists(model_right_path):
    print("Downloading RIGHT model from Google Drive...")
    gdown.download(right_url, model_right_path, quiet=False)

if not os.path.exists(model_left_path):
    print("Downloading LEFT model from Google Drive...")
    gdown.download(left_url, model_left_path, quiet=False)

print("Loading models...")
model_right = tf.keras.models.load_model(model_right_path)
model_left = tf.keras.models.load_model(model_left_path)
print("Models loaded successfully!")
# update end for .keras

with open('preprocessing/right/class_indices.json', 'r') as f:
    class_indices_right = json.load(f)

with open('preprocessing/right/preprocessing_info.json', 'r') as f:
    preprocessing_info_right = json.load(f)

with open('preprocessing/left/class_indices.json', 'r') as f:
    class_indices_left = json.load(f)

with open('preprocessing/left/preprocessing_info.json', 'r') as f:
    preprocessing_info_left = json.load(f)

idx_to_class_right = {v: k for k, v in class_indices_right.items()}
idx_to_class_left = {v: k for k, v in class_indices_left.items()}

def preprocess_image(image_path, preprocessing_info):
    img = Image.open(image_path).convert('RGB')
    target_size = tuple(preprocessing_info.get('target_size', [224, 224]))
    rescale = preprocessing_info.get('rescale', 1.0)

    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array *= rescale

    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    if 'side' not in request.form:
        return jsonify({'error': 'Please specify left or right side'}), 400
    
    file = request.files['file']
    side = request.form['side'].lower()
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if side not in ['left', 'right']:
        return jsonify({'error': 'Side must be "left" or "right"'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            if side == 'right':
                model = model_right
                preprocessing_info = preprocessing_info_right
                idx_to_class = idx_to_class_right
            else:
                model = model_left
                preprocessing_info = preprocessing_info_left
                idx_to_class = idx_to_class_left
            
            img_array = preprocess_image(filepath, preprocessing_info)
            predictions = model.predict(img_array)
            
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            results = []
            
            for idx in top_indices:
                results.append({
                    'class': idx_to_class[idx],
                    'confidence': float(predictions[0][idx] * 100)
                })
            
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'side': side,
                'predictions': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
