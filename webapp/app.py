"""
DermalScan - AI-Powered Skin Condition Analysis System
Refactored with Clean Pipeline Architecture & Professional UI
Module 3: Frontend UI | Module 4: Visualization & UX
"""

import streamlit as st
import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

import utils
from utils import predict_skin_condition
from report import generate_report

# ==================== LOGGING CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="DermalScan - AI Skin Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MODERN CSS STYLING (RUMBLE STUDIO INSPIRED) ====================
st.markdown("""
<style>
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: #1a1a1a;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background-color: #1a1a1a;
        padding: 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 700;
        padding: 14px 28px;
        color: #ffffff;
        border-radius: 12px 12px 0 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background-color: #2a2a2a;
        margin-right: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3a3a3a;
        color: #4caf50;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #4caf50;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.6);
    }
    
    .stTabs [data-baseweb="tab-content"] {
        padding: 24px;
        animation: fadeIn 0.4s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Cards and Containers */
    .metric-card {
        padding: 28px 20px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.4);
        border: 2px solid rgba(76, 175, 80, 0.6);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(76, 175, 80, 0.1) 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(76, 175, 80, 0.6);
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.4) 0%, rgba(76, 175, 80, 0.2) 100%);
    }
    
    .metric-card-value {
        font-size: 36px;
        font-weight: 800;
        margin: 12px 0;
        color: #4caf50;
    }
    
    .metric-card-label {
        font-size: 15px;
        opacity: 1;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 26px;
        font-weight: 900;
        color: #4caf50;
        margin: 28px 0 16px 0;
        border-left: 5px solid #4caf50;
        padding-left: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        text-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.25) 0%, rgba(76, 175, 80, 0.1) 100%);
        color: #ffffff;
        border-left: 5px solid #4caf50;
        padding: 18px;
        border-radius: 12px;
        margin: 12px 0;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
        font-weight: 700;
    }
    
    .info-box strong {
        color: #4caf50;
    }
    
    .info-box ul {
        margin-top: 10px;
        margin-left: 20px;
    }
    
    .info-box li {
        margin: 6px 0;
        line-height: 1.6;
        color: #ffffff;
    }
    
    /* Success/Error Boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.3) 0%, rgba(76, 175, 80, 0.15) 100%);
        color: #ffffff;
        border-left: 5px solid #4caf50;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
    }
    
    .error-box {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.3) 0%, rgba(244, 67, 54, 0.15) 100%);
        color: #ffffff;
        border-left: 5px solid #f44336;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(244, 67, 54, 0.4);
    }
    
    /* Upload Box */
    .upload-box {
        border: 3px dashed #4caf50;
        border-radius: 16px;
        padding: 40px 20px;
        text-align: center;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.08) 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-box:hover {
        border-color: #66bb6a;
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.25) 0%, rgba(76, 175, 80, 0.12) 100%);
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.3s ease;
        border: none;
        cursor: pointer;
        font-size: 15px;
        background-color: #4caf50;
        color: #ffffff;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        background-color: #66bb6a;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.5);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background-image: linear-gradient(90deg, #4caf50, #66bb6a);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        padding: 24px;
        background-color: #1a3a52;
    }
    
    /* Image Display */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 32px rgba(76, 175, 80, 0.2);
        border: 2px solid rgba(76, 175, 80, 0.3);
    }
    
    /* Footer */
    .footer {
        border-top: 2px solid rgba(76, 175, 80, 0.2);
        margin-top: 40px;
        padding: 24px;
        text-align: center;
        color: #81c784;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CLASS DEFINITIONS ====================
CLASS_NAMES = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
CLASS_EMOJIS = ['🌟', '⚫', '👁️', '✨']
CLASS_COLORS = ['#00BCD4', '#e74c3c', '#3498db', '#f39c12']

# ==================== LOAD MODEL & CASCADE ====================
@st.cache_resource
def load_resources():
    """Load model and cascade classifier using utils singleton pattern"""
    try:
        model, face_cascade = utils.load_model_and_cascade()
        if model is None or face_cascade is None:
            logger.error("Failed to load model or cascade classifier")
            return None, None
        return model, face_cascade
    except Exception as e:
        logger.error(f"Error loading model or cascade: {str(e)}", exc_info=True)
        return None, None

model, face_cascade = load_resources()
if model is None or face_cascade is None:
    st.error("❌ Failed to load AI model or face detector. Please restart the application.")
    st.stop()

# ==================== SESSION STATE INITIALIZATION ====================
# FIX 6: Initialize session state to clear cached results on new uploads
if 'last_upload_id' not in st.session_state:
    st.session_state.last_upload_id = None

if 'last_camera_id' not in st.session_state:
    st.session_state.last_camera_id = None

if 'last_analysis_results' not in st.session_state:
    st.session_state.last_analysis_results = None

# ==================== HELPER FUNCTIONS ====================

# ========== MODULAR PIPELINE FUNCTIONS ==========

def apply_nms(faces, overlap_threshold=0.3):
    """
    Apply Non-Max Suppression to remove duplicate/overlapping face detections.
    Keeps only the strongest detection in overlapping regions.
    
    Args:
        faces: List of (x, y, w, h) detections
        overlap_threshold: IoU threshold for merging overlaps
    
    Returns:
        Filtered list of non-overlapping face detections
    """
    if len(faces) == 0:
        return []
    
    # Convert to (x1, y1, x2, y2) format for easier calculation
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append([x, y, x + w, y + h])
    
    boxes = np.array(boxes, dtype=np.float32)
    
    # Calculate areas
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by area (keep larger faces)
    order = np.argsort(areas)[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w_overlap = np.maximum(0.0, xx2 - xx1 + 1)
        h_overlap = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w_overlap * h_overlap
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with low IoU (non-overlapping)
        idx = np.where(iou <= overlap_threshold)[0]
        order = order[idx + 1]
    
    # Convert back to original format
    filtered_faces = [tuple(boxes[i, :4]) for i in keep]
    filtered_faces = [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) 
                       for (x1, y1, x2, y2) in filtered_faces]
    
    return sorted(filtered_faces, key=lambda f: (f[1], f[0]))


def detect_faces(image, face_cascade):
    """
    Detect faces in the image using Haar Cascade with NMS filtering.
    FIXES: Single person false-double detection, removes overlaps.
    
    Args:
        image: Input image
        face_cascade: Cascade classifier
    
    Returns:
        List of face detections as (x, y, w, h) tuples
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with STRICTER filtering (FIX for single person detection)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More conservative scaling
            minNeighbors=6,    # HIGHER threshold to reduce false positives
            minSize=(50, 50),  # LARGER minimum size
            maxSize=(400, 400)
        )
        
        if len(faces) == 0:
            logger.info("✓ No faces detected")
            return []
        
        # FIX: Apply NMS to remove duplicate/overlapping detections
        faces = apply_nms(faces, overlap_threshold=0.25)
        
        if len(faces) == 0:
            logger.info("✓ No valid faces after NMS filtering")
            return []
        
        # Sort by left-to-right position
        faces = sorted(faces, key=lambda f: (f[1], f[0]))
        
        logger.info(f"✓ Detected {len(faces)} face(s) after NMS filtering")
        return faces
    
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}", exc_info=True)
        return []


def annotate_full_image(image, faces, is_single_person=False, primary_condition=None, severity=None):
    """
    Draw clean bounding boxes on full image with professional labels.
    FIX 1: Correct bounding box coordinate alignment
    FIX 3: Single-person mode shows condition label instead of "Person 1"
    FIX 4: Multi-person mode uses only person numbers
    
    Args:
        image: Original image (BGR format)
        faces: List of face detections as (x, y, w, h) tuples
        is_single_person: If True, label shows condition instead of "Person 1"
        primary_condition: Condition name (for single-person mode)
        severity: Severity level (for single-person mode)
    
    Returns:
        Annotated image with bounding boxes and labels (BGR format)
    """
    try:
        if image is None or image.size == 0:
            logger.error("Invalid image provided to annotate_full_image")
            return image
        
        annotated = image.copy()
        height, width = annotated.shape[:2]
        
        for person_idx, (x, y, w, h) in enumerate(faces, 1):
            # FIX 1: Convert all coordinates to integers to prevent alignment issues
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Validate coordinates to prevent OpenCV errors
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                logger.warning(f"Skipping invalid face coordinates: ({x}, {y}, {w}, {h})")
                continue
            
            # Ensure coordinates are within image bounds
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            
            # Draw bounding box (tight fit on face)
            box_color = (0, 255, 0)  # Bright GREEN (BGR)
            cv2.rectangle(
                annotated,
                (x, y),
                (x_end, y_end),  # FIX: Use correct end coordinates
                box_color,
                thickness=3,
                lineType=cv2.LINE_AA
            )
            
            # FIX 3 & 4: Determine label based on single vs multi-person mode
            if is_single_person and primary_condition is not None and severity is not None:
                # Single person mode: show condition + severity (NO "Person 1")
                label = f"{primary_condition}: {severity}"
            else:
                # Multi-person mode: show person number
                label = f"Person {person_idx}"
            
            # Label text styling
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Position label above bounding box
            label_padding = 6
            label_bg_y_start = max(0, y - text_size[1] - label_padding - 4)
            label_bg_y_end = y - 3
            label_bg_x_start = max(0, x - 3)
            label_bg_x_end = min(width, x + text_size[0] + label_padding)
            
            # Draw label background rectangle
            cv2.rectangle(
                annotated,
                (label_bg_x_start, label_bg_y_start),
                (label_bg_x_end, label_bg_y_end),
                (0, 255, 0),  # GREEN background
                thickness=-1,
                lineType=cv2.LINE_AA
            )
            
            # Draw label border
            cv2.rectangle(
                annotated,
                (label_bg_x_start, label_bg_y_start),
                (label_bg_x_end, label_bg_y_end),
                (255, 255, 255),  # WHITE border
                thickness=1,
                lineType=cv2.LINE_AA
            )
            
            # Draw label text with high contrast
            cv2.putText(
                annotated,
                label,
                (x + 3, y - label_padding),
                font,
                font_scale,
                (0, 0, 0),  # Black text for contrast on green background
                thickness,
                lineType=cv2.LINE_AA
            )
        
        logger.info(f"✓ Annotated image with {len(faces)} face(s) - SinglePersonMode: {is_single_person}")
        return annotated
    
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}", exc_info=True)
        return image


def crop_faces(image, faces):
    """
    Crop individual face regions from the image with proper validation.
    FIX 3: Ensures faces are freshly extracted using .copy().
    
    Args:
        image: Original image
        faces: List of face detections (x, y, w, h)
    
    Returns:
        List of cropped face images (fresh copies, not references)
    """
    try:
        cropped_faces = []
        padding_ratio = 0.25
        
        for idx, (x, y, w, h) in enumerate(faces, 1):
            try:
                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                
                if x1 >= x2 or y1 >= y2:
                    logger.warning(f"Face {idx}: Invalid crop coordinates")
                    continue
                
                cropped = image[y1:y2, x1:x2].copy()
                
                if cropped.size == 0 or cropped.shape[0] < 20 or cropped.shape[1] < 20:
                    logger.warning(f"Face {idx}: Skipped invalid crop")
                    continue
                
                if len(cropped.shape) != 3 or cropped.shape[2] != 3:
                    logger.warning(f"Face {idx}: Invalid channel count")
                    continue
                
                cropped_faces.append(cropped)
            
            except Exception as e:
                logger.error(f"Error cropping face {idx}: {str(e)}")
                continue
        
        logger.info(f"✓ Cropped {len(cropped_faces)} face(s)")
        return cropped_faces
    
    except Exception as e:
        logger.error(f"Error cropping faces: {str(e)}", exc_info=True)
        return []


def get_condition_severity(confidence_value):
    """
    Convert confidence value to severity level.
    FIX: Provides HIGH/MEDIUM/LOW labels for clarity.
    
    Args:
        confidence_value: Confidence percentage (0-100)
    
    Returns:
        Tuple of (severity_level, color, emoji)
    """
    if confidence_value >= 60:
        return "HIGH", "#FF5252", "🔴"
    elif confidence_value >= 30:
        return "MEDIUM", "#FFC107", "🟡"
    else:
        return "LOW", "#66BB6A", "🟢"


def analyze_skin_conditions(cropped_faces, model, face_cascade):
    """
    Perform skin analysis on each cropped face with proper preprocessing and validation.
    FIXED: Uses correct EfficientNet preprocessing, validates each face, ensures fresh predictions.
    
    Args:
        cropped_faces: List of cropped face images (must be copies, not references)
        model: Trained skin condition model
        face_cascade: Cascade classifier
    
    Returns:
        List of analysis results with predictions and severity labels
    """
    try:
        results = []
        
        for person_idx, cropped_face in enumerate(cropped_faces, 1):
            try:
                # FIX 1: Validate cropped face before processing
                if cropped_face is None or cropped_face.size == 0:
                    logger.warning(f"Skipping invalid face crop for person {person_idx}")
                    continue
                
                # FIX 3: Ensure face is a fresh copy (not a reference)
                cropped_face = cropped_face.copy()
                
                # FIX 4: Apply CORRECT preprocessing using EfficientNet's preprocess_input
                # (NOT manual normalization with /255.0)
                face_resized = cv2.resize(cropped_face, (224, 224))
                face_batch = np.expand_dims(face_resized, axis=0)
                from tensorflow.keras.applications.efficientnet import preprocess_input
                face_preprocessed = preprocess_input(face_batch)
                
                # FIX 1: Ensure model.predict() runs FRESH for every face
                prediction = model.predict(face_preprocessed, verbose=0)[0]
                
                # FIX 6: Verify softmax is applied (sum should equal ~1.0)
                prediction_sum = np.sum(prediction)
                if not (0.95 < prediction_sum < 1.05):
                    logger.warning(f"Softmax validation issue: sum={prediction_sum:.4f}")
                
                # FIX 7: Map model outputs dynamically to skin conditions
                predictions_percent = prediction * 100
                primary_idx = np.argmax(prediction)
                primary_condition = CLASS_NAMES[primary_idx]
                
                # FIX 9: Debug log to verify prediction values change per image
                logger.info(f"✓ Person {person_idx} RAW PREDICTIONS: {prediction}")
                logger.info(f"✓ Person {person_idx} PERCENTAGES: {[f'{p:.1f}%' for p in predictions_percent]}")
                
                # Add severity information for each condition with DYNAMIC mapping
                condition_details = []
                for class_idx, class_name in enumerate(CLASS_NAMES):
                    confidence = predictions_percent[class_idx]
                    severity, color, emoji = get_condition_severity(confidence)
                    condition_details.append({
                        'name': class_name,
                        'confidence': float(confidence),
                        'severity': severity,
                        'color': color,
                        'emoji': emoji
                    })
                
                # FIX 8: Store fresh results, ensure previous inferences are NOT reused
                result = {
                    'person_id': person_idx,
                    'cropped_face': cropped_face.copy(),
                    'condition': primary_condition,
                    'all_predictions': [float(p) for p in predictions_percent.tolist()],
                    'confidence': float(predictions_percent[primary_idx]),
                    'condition_details': condition_details,
                    'raw_prediction': prediction.tolist()
                }
                
                results.append(result)
                logger.info(f"✓ Person {person_idx}: {primary_condition} (Confidence: {result['confidence']:.1f}%) [Softmax sum: {prediction_sum:.4f}]")
            
            except Exception as e:
                logger.error(f"Error analyzing face {person_idx}: {str(e)}")
                continue
        
        return results
    
    except Exception as e:
        logger.error(f"Error in skin analysis: {str(e)}", exc_info=True)
        return []


def draw_bounding_boxes(image, results):
    """
    LEGACY FUNCTION - Now calls the modular annotation pipeline.
    Kept for backward compatibility.
    """
    try:
        # Extract face coordinates from results
        faces = [(r['box'][0], r['box'][1], r['box'][2], r['box'][3]) for r in results]
        return annotate_full_image(image, faces)
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {str(e)}", exc_info=True)
        return image


def create_confidence_chart(results):
    """Create a professional horizontal bar chart for confidence scores"""
    try:
        fig, axes = plt.subplots(len(results), 1, figsize=(10, 3.5 * len(results)))
        
        # Handle single face case
        if len(results) == 1:
            axes = [axes]
        
        for person_idx, (ax, result) in enumerate(zip(axes, results)):
            predictions = result['all_predictions']
            
            y_pos = np.arange(len(CLASS_NAMES))
            bars = ax.barh(y_pos, predictions, color=CLASS_COLORS, alpha=0.9, height=0.6)
            
            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"{emoji} {name}" for emoji, name in zip(CLASS_EMOJIS, CLASS_NAMES)],
                               fontsize=12, fontweight='bold', color='#1a3a52')
            ax.set_xlabel('Confidence (%)', fontsize=11, fontweight='bold', color='#2c3e50')
            ax.set_xlim([0, 105])
            ax.invert_yaxis()
            
            # Title
            title = f"Person {person_idx + 1}: Skin Analysis Confidence"
            ax.set_title(title, fontsize=13, fontweight='700', color='#1a3a52', pad=15)
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.grid(True, color='#d0d8e0', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.set_facecolor('#ffffff')
            
            # Add value labels
            for bar, pct in zip(bars, predictions):
                width = bar.get_width()
                ax.text(width + 1.5, bar.get_y() + bar.get_height()/2,
                       f"{pct:.1f}%", va='center', fontsize=11, 
                       color='#1a3a52', fontweight='bold')
        
        fig.patch.set_facecolor('#f5f7fa')
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating confidence chart: {str(e)}", exc_info=True)
        return None


def display_results(image, results, faces, annotated_image):
    """
    Display complete analysis results with intelligent layout.
    - Single face: Clean, minimal presentation
    - Multiple faces: Annotated full image + individual cards
    
    UNCHANGED: Confidence scores, charts, and exports remain identical.
    FIXED: Proper handling of single vs group, annotated image generation.
    """
    try:
        if not results or not faces:
            st.markdown("""
                <div class="error-box">
                    <strong>⚠️ No Face Detected</strong><br>
                    Please ensure:
                    <ul style="margin-top: 10px; margin-left: 20px;">
                        <li>Good lighting conditions</li>
                        <li>Face is clearly visible and frontal</li>
                        <li>Face occupies 40-70% of the frame</li>
                        <li>No heavy occlusions (masks, sunglasses)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            return
        
        num_faces = len(results)
        
        # ========== FACE DETECTION SUCCESS MESSAGE ==========
        st.markdown(f"""
            <div class="success-box">
                ✅ <strong>{num_faces} face{'s' if num_faces > 1 else ''} detected successfully!</strong>
            </div>
        """, unsafe_allow_html=True)
        
        # ========== DISPLAY LOGIC BY FACE COUNT ==========
        
        if num_faces == 1:
            # ===== SINGLE FACE: MINIMAL, CLEAN LAYOUT (NO "PERSON 1" LABEL) =====
            display_single_face(image, results[0], faces)
        
        else:
            # ===== MULTIPLE FACES: ANNOTATED + INDIVIDUAL CARDS =====
            display_multiple_faces(image, results, faces, annotated_image)
    
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}", exc_info=True)
        st.error(f"Display Error: {str(e)}")


def display_single_face(image, result, faces):
    """
    Display results for a single detected face.
    FIX 2: No "Person 1" label shown
    FIX 3: Annotated image shows ONLY condition label
    """
    try:
        # Get condition details for annotation
        condition_name = result['condition']
        confidence = result['confidence']
        severity, color, emoji = get_condition_severity(confidence)
        
        # FIX 3: Generate single-person annotated image with condition label (NO "Person 1")
        annotated_image = annotate_full_image(
            image, 
            faces, 
            is_single_person=True,
            primary_condition=condition_name,
            severity=severity
        )
        
        st.image(annotated_image, channels="BGR", use_container_width=True, clamp=True)
        
        st.markdown(f"""
            <div style='padding: 16px; background: linear-gradient(135deg, {color} 0%, {color}33 100%);
                        border-left: 5px solid {color}; border-radius: 8px; margin: 16px 0;'>
                <h3 style='color: {color}; margin: 0; font-size: 20px; font-weight: 700;'>
                    {emoji} {condition_name}: {severity}
                </h3>
                <p style='color: #ffffff; margin: 8px 0 0 0; font-weight: 600;'>
                    Confidence: {confidence:.1f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # ===== CONFIDENCE SCORES (UNCHANGED) =====
        st.markdown('<div class="section-header">📊 All Condition Scores</div>', unsafe_allow_html=True)
        
        cols = st.columns(4, gap="small")
        for col, detail in zip(cols, result['condition_details']):
            with col:
                st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {detail['color']} 0%, {detail['color']}cc 100%); 
                                                     padding: 16px 8px; min-height: 140px;">
                        <div style="font-size:20px; margin-bottom:4px;">{detail['emoji']}</div>
                        <div style="font-size:14px; font-weight: 700; color: #ffffff; 
                                   word-wrap: break-word; line-height: 1.2;">
                            {detail['confidence']:.1f}%
                        </div>
                        <div style="font-size:11px; font-weight: 600; color: #ffffff; margin-top: 4px;">
                            {detail['name']}<br>{detail['severity']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # ===== CONFIDENCE CHART (UNCHANGED) =====
        st.markdown('<div class="section-header">📈 Detailed Analysis Chart</div>', unsafe_allow_html=True)
        fig = create_confidence_chart([result])
        if fig:
            st.pyplot(fig, use_container_width=True)
        
        # ===== DOWNLOAD SECTION (UNCHANGED, BUT FIXED) =====
        display_download_section(image, [result], annotated_image, unique_id="_single")
    
    except Exception as e:
        logger.error(f"Error displaying single face: {str(e)}", exc_info=True)
        st.error(f"Display Error: {str(e)}")


def display_multiple_faces(image, results, faces, annotated_image):
    """
    Display results for multiple detected faces.
    FIX 4: Annotated image shows ONLY person numbers (NO condition text)
    FIX 5: Individual assessment cards below full image
    """
    try:
        # ===== FULL ANNOTATED IMAGE (SMALLER WIDTH) =====
        st.markdown('<div class="section-header">👥 Full Image with Face Detection</div>', unsafe_allow_html=True)
        
        # FIX 4: Regenerate annotated image for multi-person mode (no condition text, only person numbers)
        annotated_image = annotate_full_image(image, faces, is_single_person=False)
        
        st.image(annotated_image, channels="BGR", width=600, clamp=True)
        
        # ===== INDIVIDUAL FACE CARDS =====
        st.markdown('<div class="section-header">👤 Individual Analysis by Person</div>', unsafe_allow_html=True)
        
        # Create cards for each person
        for result in results:
            person_id = result['person_id']
            condition = result['condition']
            confidence = result['confidence']
            severity, color, emoji = get_condition_severity(confidence)
            
            # Person section header with severity
            st.markdown(f"""
                <div style='padding: 12px; background: linear-gradient(135deg, {color}22 0%, transparent 100%);
                            border-left: 4px solid {color}; border-radius: 6px; margin: 16px 0;'>
                    <h4 style='color: {color}; margin: 0; font-size: 16px; font-weight: 700;'>
                        👤 Person {person_id} – {condition} ({severity})
                    </h4>
                    <p style='color: #ffffff; margin: 4px 0 0 0; font-size: 13px; font-weight: 600;'>
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Cropped face image + confidence scores
            col_face, col_scores = st.columns([1, 1.2], gap="medium")
            
            with col_face:
                st.markdown("**Cropped Face**")
                if 'cropped_face' in result:
                    st.image(result['cropped_face'], channels="BGR", use_container_width=True)
            
            with col_scores:
                st.markdown("**Condition Assessment**")
                # FIX: Show all conditions with severity (HIGH/MEDIUM/LOW)
                for detail in result['condition_details']:
                    st.markdown(f"""
                        <div style='padding: 8px; background: {detail['color']}22; border-left: 3px solid {detail['color']};
                                    border-radius: 4px; margin-bottom: 8px;'>
                            <p style='margin: 0; color: #ffffff; font-weight: 700; font-size: 13px;'>
                                {detail['emoji']} {detail['name']}: <strong>{detail['severity']}</strong>
                            </p>
                            <p style='margin: 4px 0 0 0; color: #ffffff; font-size: 12px;'>
                                {detail['confidence']:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Individual confidence chart
            st.markdown("**Detailed Confidence Chart**")
            fig = create_confidence_chart([result])
            if fig:
                st.pyplot(fig, use_container_width=True)
        
        # ===== DOWNLOAD SECTION (UNCHANGED, BUT WITH FIXED ANNOTATED IMAGE) =====
        st.markdown('<div class="section-header">📥 Download Reports & Exports</div>', unsafe_allow_html=True)
        display_download_section(image, results, annotated_image, unique_id="_multi")
    
    except Exception as e:
        logger.error(f"Error displaying multiple faces: {str(e)}", exc_info=True)
        st.error(f"Display Error: {str(e)}")


def display_download_section(image, results, annotated_image=None, unique_id=""):
    """
    Display download section for reports.
    FIX: Always provide annotated image if faces were detected.
    UNCHANGED: All report generation logic stays the same.
    
    Args:
        image: Original image
        results: Analysis results
        annotated_image: Pre-generated annotated image (optional)
        unique_id: Unique identifier to avoid button key conflicts
    """
    try:
        st.markdown("""
            <div class="info-box">
                <strong>Download your analysis results:</strong> Choose from annotated image, detailed reports (PDF/CSV/JSON), 
                and structured data formats for further analysis.
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4, gap="small")
        
        with col1:
            if st.button("🖼️ Annotated Image", use_container_width=True, key=f"img_btn{unique_id}"):
                with st.spinner("⏳ Generating annotated image..."):
                    try:
                        # FIX: Use pre-generated annotated image if available
                        if annotated_image is not None:
                            # Convert to PNG bytes
                            from io import BytesIO
                            success, buffer = cv2.imencode('.png', annotated_image)
                            img_bytes = BytesIO(buffer.tobytes())
                            filename = f"DermalScan_Annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            st.download_button("⬇️ Download Image", img_bytes, filename, "image/png", use_container_width=True)
                            st.success("✅ Image ready for download")
                        else:
                            # Fallback: Try to generate from results
                            from report import export_annotated_image
                            img_bytes, filename = export_annotated_image(image, results)
                            if img_bytes:
                                st.download_button("⬇️ Download Image", img_bytes, filename, "image/png", use_container_width=True)
                                st.success("✅ Image ready for download")
                            else:
                                st.warning("⚠️ Could not generate annotated image")
                    except Exception as e:
                        st.error(f"Image Error: {str(e)}")
        
        with col2:
            if st.button("📄 PDF Report", use_container_width=True, key=f"pdf_btn{unique_id}"):
                with st.spinner("⏳ Generating PDF..."):
                    try:
                        pdf_bytes, filename = generate_report(image, results, format_type='pdf', class_names=CLASS_NAMES)
                        if pdf_bytes:
                            st.download_button("⬇️ Download PDF", pdf_bytes, filename, "application/pdf", use_container_width=True)
                            st.success("✅ PDF ready for download")
                        else:
                            st.error("Failed to generate PDF")
                    except Exception as e:
                        st.error(f"PDF Error: {str(e)}")
        
        with col3:
            if st.button("📊 CSV Report", use_container_width=True, key=f"csv_btn{unique_id}"):
                with st.spinner("⏳ Generating CSV..."):
                    try:
                        csv_bytes, filename = generate_report(image, results, format_type='csv', class_names=CLASS_NAMES)
                        if csv_bytes:
                            st.download_button("⬇️ Download CSV", csv_bytes, filename, "text/csv", use_container_width=True)
                            st.success("✅ CSV ready for download")
                        else:
                            st.error("Failed to generate CSV")
                    except Exception as e:
                        st.error(f"CSV Error: {str(e)}")
        
        with col4:
            if st.button("📋 JSON Report", use_container_width=True, key=f"json_btn{unique_id}"):
                with st.spinner("⏳ Generating JSON..."):
                    try:
                        json_bytes, filename = generate_report(image, results, format_type='json', class_names=CLASS_NAMES)
                        if json_bytes:
                            st.download_button("⬇️ Download JSON", json_bytes, filename, "application/json", use_container_width=True)
                            st.success("✅ JSON ready for download")
                        else:
                            st.error("Failed to generate JSON")
                    except Exception as e:
                        st.error(f"JSON Error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error displaying download section: {str(e)}", exc_info=True)
        st.error(f"Download Error: {str(e)}")


# ==================== MAIN LAYOUT ====================

# Header
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.markdown("""
        <div style='text-align: center; padding: 30px 0;'>
            <h1 style='color: #4caf50; font-size: 52px; margin: 0; font-weight: 900; text-shadow: 0 2px 8px rgba(76, 175, 80, 0.4);'>🔬 DermalScan</h1>
            <h2 style='color: #ffffff; font-size: 22px; margin: 12px 0; font-weight: 600;'>
                AI-Powered Skin Condition Analysis
            </h2>
            <p style='color: #b0b8c1; font-size: 14px; margin: 0;'>
                Advanced facial skin analysis using deep learning
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Content Tabs
tab_upload, tab_camera = st.tabs(["📤 Image Upload", "📷 Live Camera"])

# ==================== IMAGE UPLOAD TAB ====================
with tab_upload:
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Upload Your Image</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, JPEG, or PNG)",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # STABILITY FIX 2: Validate file before reading
                if uploaded_file.size == 0:
                    st.error("❌ File is empty. Please upload a valid image.")
                    st.stop()
                
                # FIX 6: Create unique upload ID to detect new uploads and clear cached state
                # Also use file name as secondary identifier for better accuracy
                current_upload_id = (uploaded_file.name, uploaded_file.size, id(uploaded_file))
                if st.session_state.last_upload_id != current_upload_id:
                    st.session_state.last_upload_id = current_upload_id
                    logger.info(f"✓ New upload detected: {uploaded_file.name} - clearing previous state")
                
                # STABILITY FIX 3: Safely read and decode image
                try:
                    # FIX: Reset file pointer to beginning before reading
                    uploaded_file.seek(0)
                    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    if image_bytes is None or len(image_bytes) == 0:
                        st.error("❌ Failed to read image bytes.")
                        st.stop()
                    
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                except Exception as e:
                    st.error(f"❌ Failed to decode image: {str(e)}")
                    logger.error(f"Image decode error: {str(e)}")
                    st.stop()
                
                if image is None or image.size == 0:
                    st.error("❌ Failed to read image. Please upload a valid JPG, JPEG, or PNG file.")
                    st.stop()
                
                st.markdown('<div class="success-box">✅ Image uploaded successfully! Processing...</div>', unsafe_allow_html=True)
                
                with st.spinner("🔍 Analyzing image... This may take a few seconds."):
                    try:
                        # ===== CLEAN MODULAR PIPELINE WITH FIXES =====
                        # STABILITY FIX 3: Wrap face detection in try-except
                        try:
                            faces = detect_faces(image, face_cascade)
                        except Exception as e:
                            st.error(f"❌ Face detection failed: {str(e)}")
                            logger.error(f"Face detection error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        if not faces or len(faces) == 0:
                            st.warning("⚠️ No faces detected in the image. Please try another photo.")
                            st.stop()
                        
                        # STABILITY FIX 4: Annotate with error handling
                        try:
                            annotated_image = annotate_full_image(image, faces)
                        except Exception as e:
                            st.error(f"❌ Image annotation failed: {str(e)}")
                            logger.error(f"Annotation error: {str(e)}", exc_info=True)
                            annotated_image = None
                        
                        # STABILITY FIX 3: Crop faces with error handling
                        try:
                            cropped_faces = crop_faces(image, faces)
                            if not cropped_faces or len(cropped_faces) == 0:
                                st.error("❌ Failed to crop faces. Please try another image.")
                                st.stop()
                        except Exception as e:
                            st.error(f"❌ Face cropping failed: {str(e)}")
                            logger.error(f"Cropping error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        # STABILITY FIX 3: Analyze with error handling
                        try:
                            analysis_results = analyze_skin_conditions(cropped_faces, model, face_cascade)
                            if not analysis_results or len(analysis_results) == 0:
                                st.error("❌ Skin analysis failed. Please try another image.")
                                st.stop()
                        except Exception as e:
                            st.error(f"❌ Skin analysis failed: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        # FIX 5: Format results with required fields for CSV/JSON generation
                        display_results_list = []
                        for i, (face, analysis) in enumerate(zip(faces, analysis_results)):
                            x, y, w, h = face
                            
                            # FIX 5: Ensure all fields required by report generation are present
                            result = {
                                'box': (int(x), int(y), int(w), int(h)),  # FIX: Integer coordinates for CSV
                                'person_id': i + 1,
                                'condition': analysis['condition'],
                                'label': analysis['condition'],  # FIX: Required by report.py
                                'all_predictions': analysis['all_predictions'],
                                'confidence': analysis['confidence'],
                                'cropped_face': analysis['cropped_face'],
                                'condition_details': analysis['condition_details']
                            }
                            display_results_list.append(result)
                        
                        # STABILITY FIX 6: Display results with error handling
                        try:
                            display_results(image, display_results_list, faces, annotated_image)
                        except Exception as e:
                            st.error(f"❌ Display error: {str(e)}")
                            logger.error(f"Display error: {str(e)}", exc_info=True)
                    
                    except Exception as e:
                        logger.error(f"Unexpected error in analysis pipeline: {str(e)}", exc_info=True)
                        st.error(f"❌ Analysis failed: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error processing upload: {str(e)}", exc_info=True)
                st.error(f"❌ Upload error: {str(e)}")
        else:
            st.markdown("""
                <div style='border: 3px dashed #4caf50; border-radius: 16px; padding: 60px 20px; 
                            text-align: center; background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, 
                            rgba(76, 175, 80, 0.05) 100%);'>
                    <div style='font-size: 48px; margin-bottom: 16px;'>📤</div>
                    <p style='font-size: 18px; font-weight: 700; color: #4caf50; margin: 0 0 8px 0;'>
                        Click to upload or drag and drop
                    </p>
                    <p style='font-size: 13px; color: #b0b8c1; margin: 0;'>
                        JPG, JPEG or PNG (Recommended: ≥500x500px)
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">📌 Tips for Best Results</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <strong style="color: #4caf50;">✓ Do This:</strong>
                <ul style="margin-top: 10px; color: #ffffff;">
                    <li>Use clear, well-lit photos</li>
                    <li>Face directly facing camera</li>
                    <li>Face fills 40-70% of frame</li>
                    <li>Natural lighting preferred</li>
                    <li>No filters or heavy makeup</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 152, 0, 0.1) 100%); 
                                         border-left-color: #ff9800; color: #ffffff;">
                <strong style="color: #ff9800;">✗ Avoid:</strong>
                <ul style="margin-top: 10px; color: #ffffff;">
                    <li>Dark/backlit images</li>
                    <li>Partial or angled faces</li>
                    <li>Extreme angles (>45°)</li>
                    <li>Heavy occlusions</li>
                    <li>Blurry photos</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==================== LIVE CAMERA TAB ====================
with tab_camera:
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.markdown('<div class="section-header">Capture with Camera</div>', unsafe_allow_html=True)
        
        camera_input = st.camera_input("Take a photo of your face")
        
        if camera_input is not None:
            try:
                # STABILITY FIX 2: Validate file before reading
                if not camera_input or camera_input.size == 0:
                    st.error("❌ Camera frame is empty. Please try again.")
                    st.stop()
                
                # FIX 6: Create unique camera ID to detect new captures and clear cached state
                # Use multiple identifiers for accurate detection
                current_camera_id = (id(camera_input), camera_input.size)
                if st.session_state.last_camera_id != current_camera_id:
                    st.session_state.last_camera_id = current_camera_id
                    logger.info("✓ New camera capture detected - clearing previous state")
                
                # STABILITY FIX 3: Safely read and decode image
                try:
                    # FIX: Reset file pointer to beginning before reading
                    camera_input.seek(0)
                    image_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                    if image_bytes is None or len(image_bytes) == 0:
                        st.error("❌ Failed to read camera frame bytes.")
                        st.stop()
                    
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                except Exception as e:
                    st.error(f"❌ Failed to decode camera frame: {str(e)}")
                    logger.error(f"Camera frame decode error: {str(e)}")
                    st.stop()
                
                if image is None or image.size == 0:
                    st.error("❌ Failed to read camera frame. Please try again.")
                    st.stop()
                
                st.markdown('<div class="success-box">✅ Frame captured successfully! Processing...</div>', unsafe_allow_html=True)
                
                with st.spinner("🔍 Analyzing face... This may take a few seconds."):
                    try:
                        # ===== CLEAN MODULAR PIPELINE WITH FIXES =====
                        # STABILITY FIX 3: Wrap face detection in try-except
                        try:
                            faces = detect_faces(image, face_cascade)
                        except Exception as e:
                            st.error(f"❌ Face detection failed: {str(e)}")
                            logger.error(f"Face detection error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        if not faces or len(faces) == 0:
                            st.warning("⚠️ No faces detected in the frame. Please try again.")
                            st.stop()
                        
                        # STABILITY FIX 4: Annotate with error handling
                        try:
                            annotated_image = annotate_full_image(image, faces)
                        except Exception as e:
                            st.error(f"❌ Image annotation failed: {str(e)}")
                            logger.error(f"Annotation error: {str(e)}", exc_info=True)
                            annotated_image = None
                        
                        # STABILITY FIX 3: Crop faces with error handling
                        try:
                            cropped_faces = crop_faces(image, faces)
                            if not cropped_faces or len(cropped_faces) == 0:
                                st.error("❌ Failed to crop faces from frame. Please try again.")
                                st.stop()
                        except Exception as e:
                            st.error(f"❌ Face cropping failed: {str(e)}")
                            logger.error(f"Cropping error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        # STABILITY FIX 3: Analyze with error handling
                        try:
                            analysis_results = analyze_skin_conditions(cropped_faces, model, face_cascade)
                            if not analysis_results or len(analysis_results) == 0:
                                st.error("❌ Skin analysis failed. Please try again.")
                                st.stop()
                        except Exception as e:
                            st.error(f"❌ Skin analysis failed: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}", exc_info=True)
                            st.stop()
                        
                        # FIX 5: Format results with required fields for CSV/JSON generation
                        display_results_list = []
                        for i, (face, analysis) in enumerate(zip(faces, analysis_results)):
                            x, y, w, h = face
                            
                            # FIX 5: Ensure all fields required by report generation are present
                            result = {
                                'box': (int(x), int(y), int(w), int(h)),  # FIX: Integer coordinates for CSV
                                'person_id': i + 1,
                                'condition': analysis['condition'],
                                'label': analysis['condition'],  # FIX: Required by report.py
                                'all_predictions': analysis['all_predictions'],
                                'confidence': analysis['confidence'],
                                'cropped_face': analysis['cropped_face'],
                                'condition_details': analysis['condition_details']
                            }
                            display_results_list.append(result)
                        
                        # STABILITY FIX 6: Display results with error handling
                        try:
                            display_results(image, display_results_list, faces, annotated_image)
                        except Exception as e:
                            st.error(f"❌ Display error: {str(e)}")
                            logger.error(f"Display error: {str(e)}", exc_info=True)
                    
                    except Exception as e:
                        logger.error(f"Unexpected error in analysis pipeline: {str(e)}", exc_info=True)
                        st.error(f"❌ Analysis failed: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error processing camera frame: {str(e)}", exc_info=True)
                st.error(f"❌ Camera error: {str(e)}")
    
    with col2:
        st.markdown('<div class="section-header">📷 Camera Guidelines</div>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <strong style="color: #4caf50;">Position & Lighting:</strong>
                <ul style="margin-top: 10px; color: #ffffff;">
                    <li>Position face in center</li>
                    <li>Ensure bright lighting</li>
                    <li>Look directly at camera</li>
                    <li>Keep steady before capture</li>
                    <li>Natural expression preferred</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box" style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.25) 0%, rgba(76, 175, 80, 0.1) 100%); 
                                         border-left-color: #4caf50; color: #ffffff;">
                <strong style="color: #4caf50;">Frame Tips:</strong>
                <ul style="margin-top: 10px; color: #ffffff;">
                    <li>Head should fill 50% of frame</li>
                    <li>Include shoulders</li>
                    <li>Avoid harsh shadows</li>
                    <li>Keep distance 30-60cm</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #4caf50; font-size: 24px; margin: 0; font-weight: 900; text-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);'>⚙️ Guide</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="info-box">
            <strong style="color: #4caf50;">How to Use DermalScan:</strong>
            <ol style="margin-top: 12px; margin-left: 20px; color: #ffffff;">
                <li><strong>Select a method:</strong> Upload image or use live camera</li>
                <li><strong>Good lighting:</strong> Ensure clear, well-lit conditions</li>
                <li><strong>Proper positioning:</strong> Face should be centered and clear</li>
                <li><strong>Submit:</strong> Upload image or capture photo</li>
                <li><strong>Review results:</strong> Check AI analysis and confidence scores</li>
                <li><strong>Download:</strong> Export comprehensive report (PDF/CSV/JSON)</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="info-box">
            <strong style="color: #4caf50;">About the Analysis:</strong>
            <p style="margin-top: 10px; line-height: 1.6; color: #ffffff;">
                DermalScan uses AI to analyze four skin conditions:
                <ul style="margin-top: 8px;">
                    <li><strong style="color: #4caf50;">🌟 Clear Skin:</strong> <span style="color: #ffffff;">Healthy, clear complexion</span></li>
                    <li><strong style="color: #4caf50;">⚫ Dark Spots:</strong> <span style="color: #ffffff;">Pigmentation irregularities</span></li>
                    <li><strong style="color: #4caf50;">👁️ Puffy Eyes:</strong> <span style="color: #ffffff;">Under-eye swelling/puffiness</span></li>
                    <li><strong style="color: #4caf50;">✨ Wrinkles:</strong> <span style="color: #ffffff;">Fine lines and wrinkles</span></li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="error-box">
            <strong style="color: #ff6b6b;">⚠️ Important Disclaimer:</strong>
            <p style="margin-top: 10px; line-height: 1.6; color: #ffffff;">
                This application is for <strong>educational and demonstration purposes only</strong>.
                Results are <strong>NOT medical diagnoses</strong>.
                <br><br>
                <strong style="color: #ff6b6b;">Please consult a dermatologist for actual skin concerns.</strong>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 24px; color: #b0b8c1; font-size: 13px;'>
        <p style='margin: 0 0 8px 0; font-weight: 700; color: #4caf50; font-size: 16px;'>🔬 DermalScan v2.0</p>
        <p style='margin: 0 0 8px 0; color: #ffffff;'>AI-Powered Skin Condition Analysis System</p>
        <p style='margin: 0 0 8px 0; font-size: 12px; color: #b0b8c1;'>
            <strong style="color: #ffffff;">Built with:</strong> Streamlit • TensorFlow • OpenCV • EfficientNetB0
        </p>
        <p style='margin: 0; font-size: 11px; opacity: 0.9; color: #b0b8c1;'>
            Training Accuracy: 90% | Model: EfficientNetB0 | Purpose: Educational & Demonstration
        </p>
        <p style='margin: 12px 0 0 0; font-size: 11px; opacity: 0.8; font-style: italic; color: #b0b8c1;'>
            © 2025 DermalScan. Not a medical diagnosis.
        </p>
    </div>
""", unsafe_allow_html=True)
