"""
Module 6: Backend Pipeline for Model Inference
DermalScan - AI Skin Condition Analysis

This module handles:
- Model and Haar Cascade loading (singleton pattern)
- Face detection using Haar Cascade
- Face preprocessing for EfficientNet (224x224, normalization)
- Skin condition prediction
- Comprehensive logging of inference results
- Return structured results with all class predictions
"""

import cv2
import numpy as np
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== GLOBAL VARIABLES (SINGLETON PATTERN) ====================
_model = None
_face_cascade = None

# Class names for skin conditions
CLASS_NAMES = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']

# ==================== MODEL & CASCADE LOADING ====================

def load_model_and_cascade():
    """
    Load model and Haar Cascade classifier (only once, on first call).
    Uses singleton pattern to avoid reloading.
    
    Returns:
        tuple: (model, face_cascade) or (None, None) if loading fails
    """
    global _model, _face_cascade
    
    if _model is not None and _face_cascade is not None:
        logger.info("✓ Model and Cascade already loaded (using cached version)")
        return _model, _face_cascade
    
    try:
        # Get project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(project_root, 'model', 'DermalScan_EfficientNetB0_90Percent.h5')
        cascade_path = os.path.join(project_root, 'haarcascade', 'haarcascade_frontalface_default.xml')
        
        # Validate paths exist
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None, None
        if not os.path.exists(cascade_path):
            logger.error(f"❌ Cascade file not found: {cascade_path}")
            return None, None
        
        # Load model
        logger.info(f"📦 Loading model from: {model_path}")
        _model = load_model(model_path)
        logger.info("✓ Model loaded successfully")
        
        # Load Haar Cascade
        logger.info(f"📦 Loading Haar Cascade from: {cascade_path}")
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if _face_cascade.empty():
            logger.error("❌ Failed to load Haar Cascade classifier")
            return None, None
        
        logger.info("✓ Haar Cascade loaded successfully")
        return _model, _face_cascade
        
    except Exception as e:
        logger.error(f"❌ Error during model/cascade loading: {str(e)}", exc_info=True)
        return None, None


# ==================== FACE DETECTION ====================

def detect_faces(image, face_cascade):
    """
    Detect faces in image using Haar Cascade classifier.
    
    Args:
        image (ndarray): Input image (BGR format from OpenCV)
        face_cascade: OpenCV CascadeClassifier object
        
    Returns:
        list: List of face bounding boxes as (x, y, w, h) tuples
    """
    try:
        # Convert to grayscale for Haar Cascade detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with multi-scale approach
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,        # Image size reduction at each level
            minNeighbors=5,         # Quality threshold for detection
            minSize=(60, 60),       # Minimum face size
            maxSize=(400, 400)      # Maximum face size for efficiency
        )
        
        num_faces = len(faces)
        if num_faces > 0:
            logger.info(f"✓ Face Detection: {num_faces} face(s) detected")
            for idx, (x, y, w, h) in enumerate(faces, 1):
                logger.info(f"  Face {idx}: Box=({x}, {y}, {w}, {h})")
        else:
            logger.warning("⚠️ Face Detection: No faces detected in image")
        
        return list(faces)
        
    except Exception as e:
        logger.error(f"❌ Error during face detection: {str(e)}", exc_info=True)
        return []


# ==================== FACE PREPROCESSING ====================

def preprocess_face(face_image):
    """
    Preprocess face image for EfficientNet model.
    
    Steps:
    1. Resize to 224x224 (EfficientNet input size)
    2. Normalize using EfficientNet preprocessing
    3. Add batch dimension
    
    Args:
        face_image (ndarray): Cropped face image (BGR format)
        
    Returns:
        ndarray: Preprocessed face ready for model prediction
    """
    try:
        # Validate face image
        if face_image is None or face_image.size == 0:
            logger.error("❌ Invalid face image: empty or None")
            return None
        
        if len(face_image.shape) != 3:
            logger.error(f"❌ Invalid face image shape: {face_image.shape} (expected 3D)")
            return None
        
        # Resize to EfficientNet input size
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Add batch dimension (model expects (batch, height, width, channels))
        face_batch = np.expand_dims(face_resized, axis=0)
        
        # Apply EfficientNet preprocessing (scaling and normalization)
        face_preprocessed = preprocess_input(face_batch)
        
        logger.debug("✓ Face preprocessing completed (224x224, normalized)")
        return face_preprocessed
        
    except Exception as e:
        logger.error(f"❌ Error during face preprocessing: {str(e)}", exc_info=True)
        return None


# ==================== MODEL PREDICTION ====================

def predict_skin_condition_single(preprocessed_face, model):
    """
    Run model prediction on preprocessed face.
    
    Args:
        preprocessed_face (ndarray): Preprocessed face image
        model: Loaded Keras/TensorFlow model
        
    Returns:
        ndarray: Predictions array with probabilities for all 4 classes
    """
    try:
        # Get model predictions (raw probabilities)
        predictions = model.predict(preprocessed_face, verbose=0)[0]
        
        logger.debug(f"✓ Predictions: {predictions}")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        return None


# ==================== MAIN INFERENCE FUNCTION ====================

def predict_skin_condition(image, model, face_cascade, class_names=CLASS_NAMES):
    """
    Main inference pipeline:
    1. Detect faces using Haar Cascade
    2. Preprocess each detected face
    3. Run prediction for all 4 skin conditions
    4. Return structured results with all predictions
    
    Args:
        image (ndarray): Input image (BGR format from OpenCV or Streamlit)
        model: Loaded EfficientNet model
        face_cascade: Loaded Haar Cascade classifier
        class_names (list): Names of the 4 skin condition classes
        
    Returns:
        list: List of dictionaries, each containing:
            {
                "box": (x, y, w, h),                    # Bounding box coordinates
                "label": str,                            # Top predicted class
                "confidence": float,                     # Confidence % of top class
                "all_predictions": [float, ...],        # Percentages for all 4 classes
                "class_names": list                      # Corresponding class names
            }
            Returns empty list if no faces detected or error occurs.
    """
    
    results = []
    
    logger.info("=" * 60)
    logger.info("🔍 STARTING INFERENCE PIPELINE")
    logger.info("=" * 60)
    
    # Validate inputs
    if image is None or image.size == 0:
        logger.error("❌ Invalid image provided (empty or None)")
        return results
    
    if model is None or face_cascade is None:
        logger.error("❌ Model or Cascade classifier not loaded")
        return results
    
    # Step 1: Detect faces
    logger.info("\n[STEP 1] Face Detection")
    faces = detect_faces(image, face_cascade)
    
    if not faces:
        logger.warning("⚠️ No faces detected - returning empty results")
        logger.info("=" * 60)
        return results
    
    # Step 2-3: Process each detected face
    logger.info(f"\n[STEP 2-3] Processing {len(faces)} face(s)")
    for face_idx, (x, y, w, h) in enumerate(faces, 1):
        try:
            logger.info(f"\n  ▶ Face {face_idx}/{len(faces)}")
            
            # Validate and safely crop face region with bounds checking
            height, width = image.shape[:2]
            x_start = max(0, int(x))
            y_start = max(0, int(y))
            x_end = min(width, int(x + w))
            y_end = min(height, int(y + h))
            
            # Check if crop is valid
            if x_end <= x_start or y_end <= y_start:
                logger.warning(f"  ⚠️ Face {face_idx}: Invalid bounding box after bounds checking, skipping")
                continue
            
            face_crop = image[y_start:y_end, x_start:x_end]
            
            # Check if crop is too small
            if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                logger.warning(f"  ⚠️ Face {face_idx}: Cropped face too small, skipping")
                continue
            
            # Preprocess
            preprocessed = preprocess_face(face_crop)
            if preprocessed is None:
                logger.warning(f"  ⚠️ Face {face_idx} preprocessing failed, skipping")
                continue
            
            # Predict
            predictions = predict_skin_condition_single(preprocessed, model)
            if predictions is None:
                logger.warning(f"  ⚠️ Face {face_idx} prediction failed, skipping")
                continue
            
            # Convert to percentages
            predictions_percent = predictions * 100
            
            # Get top prediction
            class_id = np.argmax(predictions_percent)
            top_label = class_names[class_id]
            top_confidence = float(predictions_percent[class_id])
            
            # Log results for this face
            logger.info(f"  ✓ Bounding Box: x={x_start}, y={y_start}, w={x_end-x_start}, h={y_end-y_start}")
            logger.info(f"  ✓ Top Prediction: {top_label} ({top_confidence:.2f}%)")
            logger.info(f"  ✓ All Predictions:")
            for class_name, percentage in zip(class_names, predictions_percent):
                logger.info(f"      • {class_name}: {percentage:.2f}%")
            
            # Append result
            results.append({
                "box": (x_start, y_start, x_end-x_start, y_end-y_start),
                "label": top_label,
                "confidence": top_confidence,
                "all_predictions": [float(p) for p in predictions_percent],
                "class_names": class_names
            })
            
        except Exception as e:
            logger.error(f"  ❌ Error processing face {face_idx}: {str(e)}", exc_info=True)
            continue
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ INFERENCE COMPLETE: {len(results)} result(s) returned")
    logger.info("=" * 60 + "\n")
    
    return results
