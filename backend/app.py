from flask import Flask, Response, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
from flask_cors import CORS
import base64
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from patchify import patchify
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class CropClassificationModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CropClassificationModel, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.features[-6:].parameters():
            param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = torch.device('cpu')
model = CropClassificationModel().to(device)
model.load_state_dict(torch.load('crop_classification_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

idx_to_class = {0: 'jute', 1: 'maize', 2: 'rice', 3: 'sugarcane', 4: 'wheat'}

@app.route('/classify', methods=['POST'])
def classify_image():
    print("Received a request for classification")
    if 'image' in request.files:
        file = request.files['image']
        print("Image received from file")
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    elif 'image_base64' in request.form:
        image_data = request.form['image_base64']
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        print("Image received from base64")
    else:
        return jsonify({'error': 'No image file or base64 data provided'}), 400

    try:
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = idx_to_class[predicted_class_index]
        print(f"Classification result: {predicted_class_name}")
        return jsonify({
            'class name': predicted_class_name
        })
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3):
    smooth = 1e-6
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_flat * y_pred_flat)
    false_neg = tf.reduce_sum(y_true_flat * (1 - y_pred_flat))
    false_pos = tf.reduce_sum((1 - y_true_flat) * y_pred_flat)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return 1 - tversky

def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def iou(y_true, y_pred):
    smooth = 1e-15
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(y_true * y_pred))
    possible_positives = tf.reduce_sum(tf.round(y_true))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(y_true * y_pred))
    predicted_positives = tf.reduce_sum(tf.round(y_pred))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.round((1 - y_true) * (1 - y_pred)))
    possible_negatives = tf.reduce_sum(tf.round(1 - y_true))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

flood_model = load_model("model.keras", custom_objects={
    'tversky_loss': tversky_loss,
    'dice_coef': dice_coef,
    'iou': iou,
    'sensitivity': sensitivity,
    'precision': precision,
    'specificity': specificity
})

cf = {
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "flat_patches_shape": (
        (256**2) // (16**2),
        16 * 16 * 3,
    )
}

def preprocess_image_for_flood_detection(image):
    """Preprocess image for flood detection model."""
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Resize image
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    
    # Normalize pixel values
    image = image / 255.0

    # Convert to patches
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = np.expand_dims(patches, axis=0)  # Add batch dimension
    
    return patches

@app.route('/detect_flood', methods=['POST'])
def detect_flood():
    print("Received a request for flood detection")
    
    try:
        # Check for image in request
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            print("Image received from file")
        elif 'image_base64' in request.form:
            image_data = request.form['image_base64']
            image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
            print("Image received from base64")
        else:
            return jsonify({'error': 'No image file or base64 data provided'}), 400
        
        # Preprocess image
        processed_image = preprocess_image_for_flood_detection(image)
        
        # Predict
        prediction = flood_model.predict(processed_image)
        prediction = np.squeeze(prediction)  # Remove batch dimension
        prediction = (prediction > 0.5).astype(np.uint8)  # Thresholding
        
        # Convert prediction to an image
        prediction_image = (prediction * 255).astype(np.uint8)
        prediction_pil = Image.fromarray(prediction_image, mode='L')  # 'L' mode for grayscale
        
        # Save to bytes buffer
        buf = io.BytesIO()
        prediction_pil.save(buf, format='PNG')
        buf.seek(0)
        
        # Determine flood status
        flood_detected = np.max(prediction) > 0
        
        # Return raw bytes directly
        return Response(buf.getvalue(), mimetype='image/png')
    
    except Exception as e:
        print(f"Error during flood detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7080) 