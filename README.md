# AICTE-Image-classification
This is a Streamlit-based web application that provides an intuitive interface for performing image classification using pre-trained deep learning models. 

The app supports two types of models:
MobileNetV2 (pre-trained on ImageNet) for classifying real-world images into 1,000 categories.
Custom CIFAR-10 model for classifying images into 10 classes, including airplane, automobile, bird, cat, dog, and more.

image_classification_app.py: Main application code.
cifar10_model.h5: Pre-trained CIFAR-10 model (ensure to place it in the root directory).
requirements.txt: Python dependencies required to run the app.

Key Features
Dual Model Support:

MobileNetV2 (ImageNet): Classifies images into 1,000 categories from the ImageNet dataset, including everyday objects, animals, and vehicles.
Custom CIFAR-10 Model: Specializes in classifying images into one of ten categories such as airplanes, automobiles, birds, cats, and more.
User-Friendly Interface:

Navigation Bar: Easily switch between MobileNetV2 and CIFAR-10 models with a sleek sidebar menu.
Real-Time Predictions: Upload an image to receive instant predictions along with confidence scores.
Educational and Practical Applications:

Ideal for learning about deep learning models and their performance.
Useful for practical applications where image classification is required.
Batch Image Upload: Supports uploading multiple images at once for batch processing and classification.

Top 3 Predictions: Displays the top 3 predicted classes with confidence scores to provide a more comprehensive output.

CSV Download: Users can download their prediction results in CSV format for further analysis or record-keeping.

Mobile Compatibility: A fully responsive design that works smoothly on both desktop and mobile devices.

Image Preprocessing: Automatically resizes and preprocesses images for model compatibility before making predictions.
