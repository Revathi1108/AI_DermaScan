# DermalScan

## AI Facial Skin Aging Detection App

**Domain: Computer Vision with Deep Learning**

---
## Dataset and Model Access

Due to GitHub file size limitations, the dataset and trained model are hosted on Google Drive.

🔗 Google Drive (Dataset + Model + Resources):  
https://drive.google.com/drive/folders/1lPJJUXUFN5l5FKDdyWRy6aueXRuILYPr?usp=sharing

The folder contains:
- Original Dataset
- Augmented Dataset
- Trained Model
- Haarcascade files
- Web application files



## 1. Project Title

**DermalScan: AI Facial Skin Aging Detection Application**

---

## 2. Project Statement

This project aims to develop a **computer vision–based system** that detects and classifies facial skin aging signs such as **wrinkles, dark spots, puffy eyes, and clear skin** using a **pretrained EfficientNetB0 deep learning model**.

The system uses **Haar Cascade** for face detection, applies preprocessing and normalization, and performs classification with percentage-based predictions.  
A **Streamlit-based web interface** allows users to upload images and visualize annotated results.

---

## 3. Objectives

- Detect and localize facial regions using computer vision
- Classify skin aging signs using a CNN model
- Display bounding boxes, labels, and confidence scores
- Provide a web-based UI for user interaction
- Export analysis results for documentation

---

## 4. Outcomes

- Successful detection of single and multiple faces
- Classification into wrinkles, dark spots, puffy eyes, and clear skin
- Percentage-based prediction output
- Annotated visual results
- Downloadable reports

---

## 5. Modules Implemented

### Module 1: Dataset Setup and Image Labeling

- Facial images categorized into four classes
- Dataset inspected and cleaned

### Module 2: Image Preprocessing and Augmentation

- Images resized to **224×224**
- Normalization applied
- Data prepared for CNN input

### Module 3: Model Training (EfficientNetB0)

- Transfer learning using pretrained EfficientNetB0
- CNN used for multi-class classification
- Model saved for inference

### Module 4: Face Detection & Prediction Pipeline

- Face detection using **OpenCV Haar Cascade**
- Cropped face regions passed to model
- Predictions generated with confidence scores

### Module 5: Web UI (Frontend)

- Streamlit-based interface
- Image upload and camera input
- Display of annotated outputs

### Module 6: Backend Pipeline

- Modular inference logic
- Integration between UI and model
- Smooth input-to-output flow

### Module 7: Export & Logging

- Download results in **PDF, CSV, and JSON**
- Annotated output generation

### Module 8: Documentation

- README documentation
- GitHub repository preparation

---

## 6. System Workflow

User Uploads Image
↓
Face Detection (Haar Cascade)
↓
Image Preprocessing
↓
EfficientNetB0 Prediction
↓
Skin Condition Classification
↓
Annotated Output
↓
Report Download

yaml
Copy code

---

## 7. Technology Stack

- Python 3.10+
- OpenCV
- TensorFlow / Keras
- EfficientNetB0
- NumPy, Pandas
- Streamlit

---

## 8. How to Run the Application

cd DermalScan_Project/webapp
pip install -r requirements.txt
streamlit run app.py
Application runs at:
http://localhost:8501

## 9. Project Structure

DermalScan_Project/
│
├── webapp/
│   ├── app.py
│   ├── utils.py
│   ├── report.py
│
├── model/
├── haarcascade/
├── README.md
## 10. Output Formats
PDF report
CSV file
JSON file

## 11. Disclaimer
This project is developed only for academic and learning purposes.
It is not intended for medical diagnosis.
## 12. Conclusion
DermalScan demonstrates an end-to-end computer vision pipeline integrated with deep learning.
The project fulfills the objectives of facial skin aging analysis using a simple, interactive, and functional web application.

