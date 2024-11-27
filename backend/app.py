import cv2
import numpy as np
from flask import Flask, Response, request, jsonify, send_file
from PIL import Image
from patchify import patchify
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
from keras.models import load_model
from keras.preprocessing import image
from flask_cors import CORS
import base64

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


cf = {
    "image_size": 256,
    "num_channels": 3,
    "patch_size": 16,
    "flat_patches_shape": (256, 48)  # Updated dynamically later
}

model = load_model('model.keras', custom_objects={"dice_loss": lambda x, y: x, "dice_coef": lambda x, y: x})

@app.route('/detect_flood', methods=['POST'])
def flood_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']

    # Read the image file into memory
    img = Image.open(io.BytesIO(img_file.read()))
    img = img.convert("RGB")  # Ensure the image is in RGB mode

    # Preprocess the image for prediction
    img = img.resize((cf["image_size"], cf["image_size"]))
    img_array = np.array(img) / 255.0

    # Patchify the image for model input
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(img_array, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, (-1, patch_shape[0] * patch_shape[1] * cf["num_channels"]))
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    # Predict the mask
    pred = model.predict(patches, verbose=0)[0]
    pred = np.reshape(pred, (cf["image_size"], cf["image_size"], 1))
    pred = (pred > 0.5).astype(np.uint8)  # Threshold prediction

    # Find edges of the flood region using Canny edge detection
    pred_edges = cv2.Canny(pred[:, :, 0] * 255, 100, 200)

    # Make edges thicker using dilation
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel (3x3 for moderate thickness)
    thicker_edges = cv2.dilate(pred_edges, kernel, iterations=3)

    # Create a blank RGB image to draw the thicker edges
    outline_mask = np.zeros((cf["image_size"], cf["image_size"], 3), dtype=np.uint8)
    outline_mask[:, :, 0] = thicker_edges  # Set the thicker edges to blue

    # Overlay the outline onto the original image
    img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
    combined_image = cv2.addWeighted(img_array, 0.9, outline_mask, 0.3, 0)
    # pred = model.predict(patches, verbose=0)[0]
    # pred = np.reshape(pred, (cf["image_size"], cf["image_size"], 1))
    # pred = (pred > 0.5).astype(np.uint8)  # Threshold prediction

    # # Create a blue mask for flood regions
    # blue_mask = np.zeros((cf["image_size"], cf["image_size"], 3), dtype=np.uint8)
    # blue_mask[:, :, 2] = pred[:, :, 0] * 255  # Set blue channel to 255 for flood regions

    # # Overlay the blue mask onto the original image
    # img_array = (img_array * 255).astype(np.uint8)  # Convert to uint8
    # combined_image = img_array.copy()

    # # Apply the blue mask only to flood regions
    # mask_indices = pred[:, :, 0] == 1
    # combined_image[mask_indices] = (0.7 * img_array[mask_indices] + 0.3 * blue_mask[mask_indices]).astype(np.uint8)

    # Save the combined image to a BytesIO object
    output = io.BytesIO()
    combined_pil_image = Image.fromarray(combined_image)
    combined_pil_image.save(output, format="PNG")
    output.seek(0)
    # Save the image in memory and send it as a response
#     output = BytesIO()
#     pred_pil_image = Image.fromarray(pred_image)
#     pred_pil_image.save(output, format="PNG")
#     output.seek(0)

    # Return the image as a response to the Flutter app
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=7080) 
