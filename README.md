# 🌱 AI-Powered Plant Disease Detection

## Overview
AI-Powered Plant Disease Detection is a deep learning application that identifies plant diseases from leaf images using a ResNet18-based Convolutional Neural Network (CNN). The system helps farmers and agricultural professionals detect diseases early and receive treatment recommendations.

## Features
- Upload plant leaf images
- Automatic disease prediction
- Confidence score display
- Treatment recommendations
- Streamlit-based user interface
- Deep learning model using PyTorch

## Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- PIL (Pillow)
- NumPy

## Dataset
PlantVillage Dataset

Supported Classes:
- Potato___Early_blight
- Potato___healthy
- Tomato___Early_blight
- Tomato___healthy

## Project Structure

Plant_Disease_Detection/
├── app.py
├── train.py
├── class_names.txt
├── requirements.txt
├── dataset/
│   └── PlantVillage/
├── model/
│   └── disease_model.pth
└── README.md

## Model
- Architecture: ResNet18
- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Transfer Learning: Yes

## How to Run

### Install Dependencies

```bash
pip install torch torchvision streamlit pillow numpy
```

### Train the Model

```bash
python train.py
```

### Run the Application

```bash
python -m streamlit run app.py
```

## Output
The application:
1. Accepts a leaf image.
2. Predicts the disease class.
3. Displays prediction confidence.
4. Provides treatment recommendations.

## Applications
- Smart Agriculture
- Disease Monitoring
- Crop Health Analysis
- Precision Farming

## Future Enhancements
- Support for additional plant diseases
- Real-time mobile deployment
- Multi-language support
- Fertilizer recommendation system
- Disease severity estimation

## Author
Aishwarya BH
B.Tech Artificial Intelligence & Data Science