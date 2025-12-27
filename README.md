---
title: VGG16 Dental X-ray Classifier
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# VGG16 Dental X-ray Classifier

A deep learning image classification web application using fine-tuned VGG16 models for dental X-ray analysis, deployed with Flask.

This system supports classification for **left** and **right** side dental X-rays using separate trained models.

---

## Features

- 🦷 Dual model system (left/right side teeth)
- 📊 Top 3 predictions with confidence scores
- ✅ Dentist validation system for continuous improvement
- 💾 Automatic data collection for model retraining
- 🔄 Models auto-download from Google Drive

---

## Installation

### Local Development

1. Clone this repository  
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the application:
```bash
   python app.py
```
4. Open: `http://localhost:5000`

### Hugging Face Spaces Deployment

This app is configured to run on Hugging Face Spaces using Docker. The models are automatically downloaded from Google Drive on first launch.

---

## Models

- **Right side model**: VGG16 fine-tuned for right side dental X-rays
- **Left side model**: VGG16 fine-tuned for left side dental X-rays

Models are hosted on Google Drive and downloaded automatically on startup.

---

## Usage

1. Select whether the X-ray is from the **left** or **right** side
2. Upload a dental X-ray image
3. View the top 3 classification predictions with confidence scores
4. Dentists can validate predictions for continuous improvement

---

## API Endpoints

- `POST /predict` - Upload image and get classification
- `POST /validate` - Submit dentist validation feedback
- `GET /get_validations` - Retrieve all validation records