import os
import json
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

# ================= CONFIGURATION =================
VALIDATION_LOG = "validation_log.json"
COLLECTED_DATA_DIR = "collected_data"

# Paths to your current BEST models
# UPDATE THESE FILENAMES to match what is currently inside your 'models' folder
MODEL_RIGHT_PATH = "models/vgg16_right_earlystop_20251216_163843.keras"
MODEL_LEFT_PATH = "models/vgg16_left_earlystop_20251217_020312.keras"

# Paths to save the NEW retrained models
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
NEW_MODEL_RIGHT_PATH = f"models/vgg16_right_retrained_{TIMESTAMP}.keras"
NEW_MODEL_LEFT_PATH = f"models/vgg16_left_retrained_{TIMESTAMP}.keras"

# Preprocessing Config
TARGET_SIZE = (224, 224)
# =================================================

def load_and_preprocess(image_path):
    """Matches the preprocessing done in app.py"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img, dtype=np.float32)
        # Rescale 1./255 matches the app logic
        img_array = img_array * (1.0/255.0)
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_class_indices(side):
    """Loads the class mapping (Name -> Number)"""
    path = f'preprocessing/{side}/class_indices.json'
    with open(path, 'r') as f:
        return json.load(f)

def retrain():
    print("\n" + "="*50)
    print("STARTING OFFLINE RETRAINING PROTOCOL")
    print("="*50)

    # 1. Check if logs exist
    if not os.path.exists(VALIDATION_LOG):
        print("No validation log found. Run the app and collect data first.")
        return

    with open(VALIDATION_LOG, "r") as f:
        logs = json.load(f)

    print(f"Loaded {len(logs)} validation entries.")

    # 2. Prepare buckets for Left vs Right data
    updates_right = {'images': [], 'labels': []}
    updates_left = {'images': [], 'labels': []}

    indices_right = get_class_indices('right')
    indices_left = get_class_indices('left')

    count_skipped = 0

    # --- HELPER: Case-Insensitive Lookup ---
    def find_class_index(name, indices):
        # 1. Try exact match (e.g. "vertical")
        if name in indices: return indices[name]
        # 2. Try lowercase match (e.g. "Vertical" -> "vertical")
        if name.lower() in indices: return indices[name.lower()]
        # 3. Try title case match (e.g. "vertical" -> "Vertical")
        if name.title() in indices: return indices[name.title()]
        return None

    # 3. Process logs and load images
    for entry in logs:
        # Filter: We only learn from mistakes that have a correction
        if entry.get('feedback') == 'wrong' and entry.get('correct_class'):
            
            image_id = entry['image_id']
            side = entry.get('side', 'right').lower() # Default to right if missing
            correct_label_name = entry['correct_class']

            # Find the actual image file
            search_pattern = os.path.join(COLLECTED_DATA_DIR, f"{image_id}_*")
            files = glob.glob(search_pattern)

            if not files:
                print(f"Image file missing for ID {image_id} - Skipping")
                count_skipped += 1
                continue
            
            # Load Image
            img_array = load_and_preprocess(files[0])
            if img_array is None: continue

            # Map text label to number using smart lookup
            if side == 'right':
                label_idx = find_class_index(correct_label_name, indices_right)
                if label_idx is not None:
                    updates_right['images'].append(img_array)
                    updates_right['labels'].append(label_idx)
                    print(f"   -> Added '{correct_label_name}' to RIGHT training set")
                else:
                    print(f"Skipped: '{correct_label_name}' not found in RIGHT indices")
            else:
                label_idx = find_class_index(correct_label_name, indices_left)
                if label_idx is not None:
                    updates_left['images'].append(img_array)
                    updates_left['labels'].append(label_idx)
                    print(f"   -> Added '{correct_label_name}' to LEFT training set")
                else:
                    print(f"Skipped: '{correct_label_name}' not found in LEFT indices")

    # 4. Retrain RIGHT Model
    if len(updates_right['images']) > 0:
        print(f"\nRetraining RIGHT model with {len(updates_right['images'])} new corrections...")
        
        # Load Model
        model = tf.keras.models.load_model(MODEL_RIGHT_PATH)
        
        # Compile with VERY LOW learning rate (Fine-Tuning)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Convert to numpy arrays
        X = np.array(updates_right['images'])
        y = np.array(updates_right['labels'])

        # Train
        model.fit(X, y, epochs=5, batch_size=4, verbose=1)
        
        # Save
        model.save(NEW_MODEL_RIGHT_PATH)
        print(f"SUCCESS: New Right Model saved to: {NEW_MODEL_RIGHT_PATH}")
    else:
        print("\nNo new data for RIGHT model.")

    # 5. Retrain LEFT Model
    if len(updates_left['images']) > 0:
        print(f"\nRetraining LEFT model with {len(updates_left['images'])} new corrections...")
        
        model = tf.keras.models.load_model(MODEL_LEFT_PATH)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        X = np.array(updates_left['images'])
        y = np.array(updates_left['labels'])

        model.fit(X, y, epochs=5, batch_size=4, verbose=1)
        
        model.save(NEW_MODEL_LEFT_PATH)
        print(f"SUCCESS: New Left Model saved to: {NEW_MODEL_LEFT_PATH}")
    else:
        print("\nNo new data for LEFT model.")

    print("\n" + "="*50)
    print("RETRAINING COMPLETE")
    print("If new models were created, update app.py to point to the new filenames!")
    print("="*50)

if __name__ == "__main__":
    retrain()