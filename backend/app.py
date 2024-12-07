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
import onnxruntime as ort
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

@app.route('/classify_crop', methods=['POST'])
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

@app.route('/detect', methods=['POST'])
def detect_flood():
    print("Received a request for flood detection")
    
    try:
        # Check for image in request
        if 'image' in request.files:
            file = request.files['image']
            original_image = Image.open(file).convert('RGB')
            print("Image received from file")
        else:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Resize input image to match model's expected input size
        image = original_image.resize((cf["image_size"], cf["image_size"]))
                
        # Preprocess image
        processed_image = preprocess_image_for_flood_detection(image)
        print("Image preprocessed for flood detection")
        
        # Predict
        prediction = flood_model.predict(processed_image)
        print("Prediction made by the model")
        prediction = np.squeeze(prediction)  # Remove batch dimension
        prediction = (prediction > 0.5).astype(np.uint8)  # Thresholding

        # Convert prediction to images
        # Predicted Mask
        predicted_mask_image = (prediction * 255).astype(np.uint8)
        predicted_mask_pil = Image.fromarray(predicted_mask_image, mode='L')  # 'L' mode for grayscale

        # Result Image with Circled Flood Areas
        result_image = np.array(original_image).copy()
        result_image = cv2.resize(result_image, (cf["image_size"], cf["image_size"]))
        
        # Find contours of flood areas
        contours, _ = cv2.findContours(predicted_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(result_image, contours, -1, (255, 0, 0), 4)
        
        result_image_pil = Image.fromarray(result_image)
        
        # Save images to byte buffers        
        predicted_mask_buf = io.BytesIO()
        predicted_mask_pil.save(predicted_mask_buf, format='PNG')
        predicted_mask_buf.seek(0)
        
        result_image_buf = io.BytesIO()
        result_image_pil.save(result_image_buf, format='PNG')
        result_image_buf.seek(0)
        
        # Encode images to base64
        predicted_mask_base64 = base64.b64encode(predicted_mask_buf.getvalue()).decode('utf-8')
        result_image_base64 = base64.b64encode(result_image_buf.getvalue()).decode('utf-8')
        
        # Return JSON with base64 encoded images
        return jsonify({
            'predicted_mask': predicted_mask_base64,
            'result_image': result_image_base64,
            'flood_detected': bool(np.max(prediction) > 0)
        })
    
    except Exception as e:
        print(f"Error during flood detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

def preprocess_sar_image(image_buffer):
    """
    Preprocess SAR image for ONNX model with explicit float32 conversion
    
    Args:
        image_buffer (bytes): Input image buffer
    
    Returns:
        numpy.ndarray: Preprocessed image tensor
    """
    # Open image
    image = Image.open(io.BytesIO(image_buffer)).convert('RGB')
    
    # Resize to 256x256
    image = image.resize((256, 256))
    
    # Convert to numpy array and explicitly set to float32
    image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [-1, 1] range (Pix2Pix style normalization)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    # Transpose to CHW format
    image_array = image_array.transpose((2, 0, 1))
    image_array = (image_array / 255.0 - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def postprocess_sar_image(output_tensor):
    """
    Postprocess the colorized SAR image output
    
    Args:
        output_tensor (numpy.ndarray): Model output tensor
    
    Returns:
        numpy.ndarray: Processed image
    """
    # Remove batch dimension
    output_image = output_tensor[0]
    
    # Transpose back to HWC
    output_image = output_image.transpose((1, 2, 0))
    
    # Denormalize
    output_image = (output_image * 0.5 + 0.5) * 255
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image

# Initialize ONNX Runtime session
try:
    sar_session = ort.InferenceSession("colorization.onnx")
    print("SAR colorization model loaded successfully")
    
    # Print input details for debugging
    input_info = sar_session.get_inputs()[0]
    print(f"Input name: {input_info.name}")
    print(f"Input type: {input_info.type}")
    print(f"Input shape: {input_info.shape}")
except Exception as e:
    print(f"Error loading SAR colorization model: {e}")
    sar_session = None

@app.route('/colorize', methods=['POST'])
def colorize_sar_image():
    """
    Endpoint for SAR image colorization
    """
    print("Received a request for SAR image colorization")
    
    if sar_session is None:
        return jsonify({'error': 'SAR colorization model not loaded'}), 500
    
    # Check for image in request
    if 'image' in request.files:
        file = request.files['image']
        image_buffer = file.read()
        print("Image received from file")
    elif request.json and 'image' in request.json:
        # If image is sent as base64 in JSON
        image_buffer = base64.b64decode(request.json['image'])
        print("Image received from base64")
    else:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Preprocess image
        input_tensor = preprocess_sar_image(image_buffer)
        
        # Prepare input for ONNX Runtime
        input_name = sar_session.get_inputs()[0].name
        
        # Run inference
        outputs = sar_session.run(None, {input_name: input_tensor})
        
        # Postprocess image
        result_image = postprocess_sar_image(outputs[0])
        
        # Convert to PIL Image
        pil_image = Image.fromarray(result_image)
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        colorized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'colorizedImage': colorized_base64
        })
    
    except Exception as e:
        print(f"Error during SAR image colorization: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7080)