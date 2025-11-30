from flask import Flask, render_template, request, jsonify 
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

print("Checking models...")

# Try to download models from Google Drive if not present
try:
    import gdown
    os.makedirs("models", exist_ok=True)

    model_right_path = "models/vgg16_fine_tuned_right.keras"
    model_left_path = "models/vgg16_fine_tuned_left.keras"

    # Replace these with your actual Google Drive file IDs
    right_url = "https://drive.google.com/uc?id=17AmUDIx_VctABo9WutOphRHkSNGw3Dxr"
    left_url = "https://drive.google.com/uc?id=1jxChWv1tRdecaPLCx45mO0BxbIOyOkft"

    if not os.path.exists(model_right_path):
        print("Downloading RIGHT model from Google Drive...")
        gdown.download(right_url, model_right_path, quiet=False)
    else:
        print("RIGHT model already exists locally")

    if not os.path.exists(model_left_path):
        print("Downloading LEFT model from Google Drive...")
        gdown.download(left_url, model_left_path, quiet=False)
    else:
        print("LEFT model already exists locally")
except ImportError:
    print("gdown not installed, using local models only")
    model_right_path = "models/vgg16_fine_tuned_right.keras"
    model_left_path = "models/vgg16_fine_tuned_left.keras"

print("Loading models...")
model_right = tf.keras.models.load_model(model_right_path)
model_left = tf.keras.models.load_model(model_left_path)
print("Models loaded successfully!")

# Load preprocessing info
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
    """Preprocess image for VGG16 model"""
    img = Image.open(image_path).convert('RGB')
    
    # Get target size from preprocessing info
    target_size = preprocessing_info.get('target_size', [224, 224])
    if isinstance(target_size, list):
        target_size = tuple(target_size)
    
    # Resize image
    img = img.resize(target_size)
    
    # Convert to array
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply rescaling from preprocessing info
    rescale = preprocessing_info.get('rescale', 1.0/255.0)
    img_array = img_array * rescale
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

# Storage for dentist validation feedback
validated_results = []

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
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing {side} side image: {filename}")
        
        # Select appropriate model and preprocessing
        if side == 'right':
            model = model_right
            preprocessing_info = preprocessing_info_right
            idx_to_class = idx_to_class_right
        else:
            model = model_left
            preprocessing_info = preprocessing_info_left
            idx_to_class = idx_to_class_left
        
        # Preprocess and predict
        print("Preprocessing image...")
        img_array = preprocess_image(filepath, preprocessing_info)
        
        print("Making prediction...")
        predictions = model.predict(img_array, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_indices:
            results.append({
                'class': idx_to_class[idx],
                'confidence': float(predictions[0][idx] * 100)
            })
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Generate unique ID for this prediction
        image_id = str(uuid.uuid4())
        
        print(f"Prediction complete! Top result: {results[0]['class']} ({results[0]['confidence']:.2f}%)")
        
        return jsonify({
            'success': True,
            'side': side,
            'predictions': results,
            'image_id': image_id
        })
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate_prediction():
    """Endpoint for dentist validation"""
    try:
        data = request.json
        image_id = data.get("image_id")
        prediction = data.get("prediction")
        feedback = data.get("feedback")  # "correct" or "wrong"
        correct_class = data.get("correct_class")  # Only present if wrong

        if not image_id or not prediction or not feedback:
            return jsonify({"error": "Missing fields"}), 400

        # Store validation
        validation_entry = {
            "image_id": image_id,
            "prediction": prediction,
            "feedback": feedback,
            "timestamp": str(np.datetime64('now'))
        }
        
        # Add correct class if dentist provided it
        if feedback == 'wrong' and correct_class:
            validation_entry["correct_class"] = correct_class

        validated_results.append(validation_entry)

        log_message = f"Validation received: {prediction} - {feedback}"
        if correct_class:
            log_message += f" (Correct: {correct_class})"
        print(log_message)
        
        return jsonify({
            "success": True, 
            "message": "Dentist validation saved!",
            "total_validations": len(validated_results)
        })
    
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_validations', methods=['GET'])
def get_validations():
    """Get all validation results"""
    return jsonify({
        "success": True,
        "total": len(validated_results),
        "validations": validated_results
    })

if __name__ == '__main__':
    # Check if running locally or on server
    import sys
    
    # For local development with ngrok
    if '--ngrok' in sys.argv:
        try:
            from pyngrok import ngrok
            
            # Get ngrok token from environment or set it here
            NGROK_TOKEN = os.environ.get('NGROK_AUTH_TOKEN', 'YOUR_NGROK_TOKEN_HERE')
            
            if NGROK_TOKEN != 'YOUR_NGROK_TOKEN_HERE':
                ngrok.set_auth_token(NGROK_TOKEN)
                public_url = ngrok.connect(5000)
                print("\n" + "="*70)
                print("✅ VGG16 CLASSIFIER IS LIVE!")
                print("="*70)
                print(f"\n🌐 PUBLIC URL (Share this link):")
                print(f"   {public_url}\n")
                print("="*70)
                print("\n⚠️  IMPORTANT:")
                print("   - Keep this terminal window open")
                print("   - When you close it, the link stops working")
                print("   - Your computer must stay connected to internet\n")
                print("="*70 + "\n")
            else:
                print("\n⚠️  To use ngrok, set NGROK_AUTH_TOKEN environment variable")
                print("   or replace 'YOUR_NGROK_TOKEN_HERE' in the code\n")
        except ImportError:
            print("\n⚠️  pyngrok not installed. Install with: pip install pyngrok")
        except Exception as e:
            print(f"\n❌ ngrok error: {e}")
            print("Running without ngrok (local only)\n")
    
    # Standard local server
    print("\n" + "="*70)
    print("🚀 Starting Flask Server")
    print("="*70)
    print(f"\n📍 Local URL: http://127.0.0.1:5000")
    print(f"📍 Network URL: http://localhost:5000")
    print("\n💡 To use ngrok for public access, run:")
    print("   python app.py --ngrok")
    print("\n" + "="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)