import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torchvision import transforms, datasets ,models
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
generator = torch.Generator().manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# Directories
root_dir = '/content/drive/MyDrive/dataset/agriculture-crop-images/crop_images'
save_dir = '/content/drive/MyDrive/dataset/agriculture-crop-images/aug'

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image
from collections import Counter
# Define a transformation to convert PIL images to tensors
transform_to_tensor = transforms.Compose([
    transforms.ToTensor()
])


# Create save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Dataset loader for the original data with transformation to convert PIL to Tensor
dataset = ImageFolder(root=root_dir, transform=transform_to_tensor)

# DataLoader to iterate through the dataset
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Augmentation transformation
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.ToTensor()
])
# Loop through the dataset and save augmented images
for idx, (image, label) in enumerate(dataloader):
    pil_image = transforms.ToPILImage()(image.squeeze(0))  # Convert tensor back to PIL image

    # Get the original image path to maintain subfolder structure
    original_image_path = dataset.imgs[idx][0]
    original_subfolder = os.path.basename(os.path.dirname(original_image_path))

    # Define save path
    save_subfolder = os.path.join(save_dir, original_subfolder)
    if not os.path.exists(save_subfolder):
        os.makedirs(save_subfolder)

    # Save augmented versions of the image
    for i in range(4):  # Create 4 augmentations
        augmented_image = augment_transform(pil_image)
        save_image_path = os.path.join(save_subfolder, f'augmented_{idx}_{i}.jpg')
        augmented_image_pil = transforms.ToPILImage()(augmented_image) # Removed the = symbol
        augmented_image_pil.save(save_image_path)

import os

def print_folder_image_counts_before_augmentation(root_dir):
    """
    Prints the number of images in each folder before augmentation.

    Args:
        root_dir (str): The root directory containing subfolders of images.
    """
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            num_images = len(os.listdir(subdir_path))
            print(f"Folder: {subdir}, Number of images: {num_images}")

class CropDatasetFromFolders(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# create dataset
Dataset = CropDatasetFromFolders(root_dir=save_dir,transform = transform)

train_dir = '/content/drive/MyDrive/dataset/agriculture-crop-images/train1'  # Folder to store training images
test_dir = '/content/drive/MyDrive/dataset/agriculture-crop-images/test1'

import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_save_dataset(original_data_dir, train_dir, test_dir, test_size=0.2):
    """
    Split the dataset into training and testing sets and save the images into different folders.

    Args:
    - original_data_dir (str): The path to the original dataset containing subfolders for each class.
    - train_dir (str): The directory where training images will be saved.
    - test_dir (str): The directory where testing images will be saved.
    - test_size (float): The proportion of the dataset to include in the test split (default 0.2).
    """
    # Create train and test directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # List all class subfolders in the original data directory
    class_folders = [folder for folder in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, folder))]

    for class_name in class_folders:
        # Paths to class subfolder in the original, train, and test directories
        class_folder = os.path.join(original_data_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        # Create class subfolders in the train and test directories
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        # List all image files in the current class folder
        image_files = [img for img in os.listdir(class_folder) if img.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

        # Split the dataset into training and testing sets (80% train, 20% test)
        train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)

        # Copy training files to the corresponding train directory
        for img_file in train_files:
            img_path = os.path.join(class_folder, img_file)
            shutil.copy(img_path, train_class_dir)

        # Copy testing files to the corresponding test directory
        for img_file in test_files:
            img_path = os.path.join(class_folder, img_file)
            shutil.copy(img_path, test_class_dir)

# Example usage
  # Root folder of your original datasetsave_dir = '/content/drive/MyDrive/dataset/agriculture-crop-images/aug'

 # Folder to store testing images

# Split and save the dataset with 80% training and 20% testing
split_and_save_dataset(save_dir, train_dir, test_dir, test_size=0.2)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

#create dataloder for train set and test set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

dataiter = iter(train_loader)
images, labels = next(dataiter)

num_images = 9

grid_dim = int(np.sqrt(num_images))

fig, axes = plt.subplots(grid_dim, grid_dim, figsize=(10, 10))


axes = axes.flatten()

for i in range(num_images):
    img = images[i]
    label = labels[i]



    img = np.transpose(img.numpy(), (1, 2, 0)) # C, H, W -> H, W, C

    axes[i].imshow(img)
    axes[i].set_title(f'Label: {label}')  # Set title
    axes[i].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()

# Load pretrained VGG16 model
pre_model = models.vgg16(pretrained=True).to('cuda')
for param in pre_model.parameters():
    param.requires_grad = False
for param in pre_model.features[-6:].parameters():
    param.requires_grad = True

classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 5)
)
pre_model.classifier = classifier

#define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model = pre_model.to(device)

num_epochs = 4
lr = 0.0001
optimizer = Adam(Model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    Model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for image, labels in train_loader:
        image = image.to(device)
        labels = labels.to(device)

        outputs = Model(image)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy*100:.2f}%")



    train_losses.append(epoch_loss)

    #-----------------------------------------------------------------------------------

    # Evaluate on the test set
    Model.eval()
    test_correct_predictions = 0
    test_total_samples = 0
    test_running_loss = 0.0

    with torch.no_grad():
        for image, labels in test_loader:
            image, labels = image.to(device), labels.to(device)

            outputs = Model(image)
            loss = criterion(outputs, labels)

            test_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_correct_predictions += (predicted == labels).sum().item()
            test_total_samples += labels.size(0)

    test_accuracy = test_correct_predictions / test_total_samples
    test_loss = test_running_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")
    print()
    print("----------------------------------------------------------------------------")


    test_losses.append(test_loss)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

test_correct_predictions = 0
test_total_samples = 0
all_predicted = []
all_labels = []

Model.eval()
with torch.no_grad():
    for image, labels in test_loader:
        image, labels = image.to(device), labels.to(device)

        outputs = Model(image)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        test_correct_predictions += (predicted == labels).sum().item()
        test_total_samples += labels.size(0)

        # Collect predictions and true labels
        all_predicted.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
test_accuracy = test_correct_predictions / test_total_samples
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix
cm = confusion_matrix(all_labels, all_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Dataset.classes)

# Plot the confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

# After collecting predictions and true labels
precision = precision_score(all_labels, all_predicted, average='weighted')
recall = recall_score(all_labels, all_predicted, average='weighted')
f1 = f1_score(all_labels, all_predicted, average='weighted')

# Print all metrics
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


incorrect = []
correct = []

# Put the model in evaluation mode
pre_model.eval()

# Go through the test dataset
for i in range(len(test_dataset)):
    image, label = test_dataset[i]
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        prediction = pre_model(image)
        y_hat = torch.argmax(prediction, dim=1).item()

    if y_hat != label:
        incorrect.append([image, y_hat, label])
    else:
        correct.append([image, y_hat, label])
print(f'Number of all samples: {len(test_dataset)}')
print(f'Number of incorrect samples: {len(incorrect)}')
print(f'Number of correct samples: {len(correct)}')


num_samples = min(len(incorrect), 5)  # Plot at most 5 incorrect samples
plt.figure(figsize=(15, 10))
for i in range(num_samples):
    image, y_hat, label = incorrect[i]
    image = image.squeeze(0).cpu().numpy()
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.axis('off')
    plt.title(f'Predicted: {y_hat}, True: {label}')

plt.tight_layout()
plt.show()


from torchsummary import summary
summary(Model, input_size=(3, 224,224))


summary(pre_model, input_size=(3, 224,224))


from google.colab import files
uploaded = files.upload()

# Get the first uploaded image's filename
image_path = list(uploaded.keys())[0]

# Preprocess the image and classify it
def classify_single_image(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')  # Load image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations and add batch dimension
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        output = model(image)  # Run image through model
        _, predicted = torch.max(output, 1)  # Get class with the highest score

    return predicted.item()
# Create a reverse dictionary to map index back to class name
idx_to_class = {v: k for k, v in Dataset.class_to_idx.items()}

# Use the function to classify the uploaded image
predicted_class_index = classify_single_image(image_path, pre_model, transform, device)

# Get the corresponding class name
predicted_class_name = idx_to_class[predicted_class_index]

# Print the predicted class name
print(f'Predicted Class Index: {predicted_class_index}')
print(f'Predicted Class Name: {predicted_class_name}')


