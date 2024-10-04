from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Define your model architecture (same as before)
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

# Load the model (explicitly on CPU)
device = torch.device('cpu')
model = CropClassificationModel().to(device)
model.load_state_dict(torch.load('crop_classification_model.pth', map_location=device))
model.eval()

# Define the transformation (same as before)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define your class mapping 
idx_to_class = {0: 'jute', 1: 'maize', 2: 'rice', 3: 'sugarcane', 4: 'wheat'}

@app.route('/classify', methods=['POST'])
def classify_image():
    print("Received a request for classification")  # Add this to log incoming requests
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    print("Image received")  # Log that the image has been received
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
        predicted_class_index = predicted.item()
        predicted_class_name = idx_to_class[predicted_class_index]
        print(f"Classification result: {predicted_class_name}")  # Log the classification result
        return jsonify({
            'class name': predicted_class_name
        })
    except Exception as e:
        print(f"Error during classification: {str(e)}")  # Log any errors during classification
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)