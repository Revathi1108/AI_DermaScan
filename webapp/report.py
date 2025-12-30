"""
Module 7: Downloadable Skin Analysis Report
DermalScan - AI Skin Condition Analysis

This module generates comprehensive PDF reports for skin analysis results:
- Supports multiple people in one image/frame
- Each person gets a separate section
- Includes cropped face images, metrics, and charts
- Generates PDF with embedded visualizations
- Supports CSV/JSON export as secondary formats
"""

import io
import os
import csv
import json
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================
CLASS_NAMES = ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
CLASS_EMOJIS = ['🌟', '⚫', '👁️', '✨']
CLASS_COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']


# ==================== CHART GENERATION ====================

def generate_confidence_chart(all_predictions, class_names=CLASS_NAMES):
    """
    Generate a matplotlib bar chart for prediction percentages.
    
    Args:
        all_predictions (list): List of prediction percentages (0-100)
        class_names (list): Names of classes
        
    Returns:
        io.BytesIO: Chart image as PNG bytes
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(class_names))
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        bars = ax.barh(y_pos, all_predictions, color=colors, alpha=0.85)
        
        # Style
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=11, fontweight='bold')
        ax.set_xlabel('Confidence Percentage (%)', fontsize=10, fontweight='bold')
        ax.set_xlim([0, 105])
        ax.invert_yaxis()
        ax.set_title('Skin Condition Analysis', fontsize=12, fontweight='bold', pad=15)
        
        # Add grid and remove spines
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, all_predictions):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close(fig)
        
        return img_bytes
        
    except Exception as e:
        logger.error(f"❌ Error generating chart: {str(e)}", exc_info=True)
        return None


def face_image_to_bytes(face_crop):
    """
    Convert face crop (numpy array) to PNG bytes for embedding in PDF.
    
    Args:
        face_crop (ndarray): Face image in BGR format
        
    Returns:
        io.BytesIO: Face image as PNG bytes
    """
    try:
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes
        
    except Exception as e:
        logger.error(f"❌ Error converting face image: {str(e)}", exc_info=True)
        return None


# ==================== PDF GENERATION ====================

def generate_pdf_report(original_image, results, class_names=CLASS_NAMES):
    """
    Generate comprehensive PDF report for skin analysis.
    
    Args:
        original_image (ndarray): Original input image (BGR)
        results (list): List of detection results from predict_skin_condition()
        class_names (list): Names of classes
        
    Returns:
        tuple: (pdf_bytes, filename) or (None, None) if generation fails
    """
    
    if not REPORTLAB_AVAILABLE:
        logger.error("❌ reportlab not available. Install with: pip install reportlab")
        return None, None
    
    if not results:
        logger.warning("⚠️ No results to generate report")
        return None, None
    
    try:
        # Create PDF document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Skin_Analysis_Report_{timestamp}.pdf"
        
        # Create in-memory PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1a3a52'),
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        person_style = ParagraphStyle(
            'PersonHeading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Build PDF content
        content = []
        
        # ===== HEADER =====
        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph("🔬 DermalScan", title_style))
        content.append(Paragraph("AI-Powered Skin Condition Analysis Report", styles['Normal']))
        content.append(Spacer(1, 0.1*inch))
        
        # Report metadata
        report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content.append(Paragraph(f"<b>Report Generated:</b> {report_time}", normal_style))
        content.append(Paragraph(f"<b>Total Faces Detected:</b> {len(results)}", normal_style))
        content.append(Spacer(1, 0.2*inch))
        
        # ===== MAIN IMAGE =====
        content.append(Paragraph("📸 Analysis Image", heading_style))
        try:
            # Resize original image for display
            height, width = original_image.shape[:2]
            scale = 4.5 / (width / 100)  # Scale to 4.5 inches width
            new_height = int(height * scale)
            resized = cv2.resize(original_image, (int(width * scale), new_height))
            
            # Convert to RGB and save to bytes
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            rl_image = RLImage(img_bytes, width=4.5*inch, height=new_height/100*inch)
            content.append(rl_image)
        except Exception as e:
            logger.warning(f"⚠️ Could not embed main image: {str(e)}")
            content.append(Paragraph("(Main image could not be embedded)", normal_style))
        
        content.append(Spacer(1, 0.3*inch))
        
        # ===== INDIVIDUAL RESULTS =====
        content.append(Paragraph("📋 Detailed Results", heading_style))
        content.append(Spacer(1, 0.1*inch))
        
        for person_idx, result in enumerate(results, 1):
            # Person section
            person_section = []
            person_section.append(Paragraph(f"Person {person_idx}", person_style))
            
            # Bounding box info
            x, y, w, h = result['box']
            person_section.append(Paragraph(
                f"<b>Bounding Box:</b> X={x}, Y={y}, Width={w}, Height={h}", 
                normal_style
            ))
            
            # Extract and embed cropped face
            try:
                face_crop = original_image[y:y+h, x:x+w]
                if face_crop.size > 0:
                    # Resize for display
                    face_resized = cv2.resize(face_crop, (200, 200))
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                    pil_face = Image.fromarray(face_rgb)
                    face_bytes = io.BytesIO()
                    pil_face.save(face_bytes, format='PNG')
                    face_bytes.seek(0)
                    
                    face_img = RLImage(face_bytes, width=1.8*inch, height=1.8*inch)
                    person_section.append(face_img)
            except Exception as e:
                logger.warning(f"⚠️ Could not embed face image for person {person_idx}: {str(e)}")
            
            person_section.append(Spacer(1, 0.15*inch))
            
            # Prediction table
            table_data = [['Skin Condition', 'Confidence']]
            all_preds = result['all_predictions']
            for class_name, confidence in zip(class_names, all_preds):
                table_data.append([class_name, f'{confidence:.2f}%'])
            
            # Top prediction highlight
            top_idx = np.argmax(all_preds)
            top_label = class_names[top_idx]
            top_conf = all_preds[top_idx]
            person_section.append(Paragraph(
                f"<b>Primary Prediction:</b> {top_label} ({top_conf:.2f}%)",
                normal_style
            ))
            
            pred_table = Table(table_data, colWidths=[3.5*inch, 1.5*inch])
            pred_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
            ]))
            person_section.append(Spacer(1, 0.1*inch))
            person_section.append(pred_table)
            
            # Generate and embed chart
            try:
                chart_bytes = generate_confidence_chart(all_preds, class_names)
                if chart_bytes:
                    chart_img = RLImage(chart_bytes, width=5*inch, height=2.5*inch)
                    person_section.append(Spacer(1, 0.15*inch))
                    person_section.append(chart_img)
            except Exception as e:
                logger.warning(f"⚠️ Could not generate chart for person {person_idx}: {str(e)}")
            
            # Keep person section together
            content.append(KeepTogether(person_section))
            
            # Page break between persons if needed
            if person_idx < len(results):
                content.append(Spacer(1, 0.3*inch))
                content.append(Paragraph("___" * 20, normal_style))
                content.append(Spacer(1, 0.3*inch))
        
        # ===== FOOTER =====
        content.append(Spacer(1, 0.3*inch))
        content.append(Paragraph("---" * 30, normal_style))
        footer_text = (
            "DermalScan v2.0 | AI-Powered Skin Analysis System<br/>"
            "Model: EfficientNetB0 (90% Accuracy) | For Educational & Demonstration Purposes<br/>"
            f"Report Generated: {report_time}"
        )
        content.append(Paragraph(footer_text, styles['Normal']))
        
        # Build PDF
        doc.build(content)
        
        # Get bytes
        pdf_buffer.seek(0)
        logger.info(f"✓ PDF report generated successfully: {filename}")
        
        return pdf_buffer, filename
        
    except Exception as e:
        logger.error(f"❌ Error generating PDF: {str(e)}", exc_info=True)
        return None, None


# ==================== CSV EXPORT ====================

def generate_csv_report(original_image, results, class_names=CLASS_NAMES):
    """
    Generate CSV report for skin analysis results in Infosys-compliant format.
    
    CSV Columns:
    - image_name
    - face_id
    - clear_skin_percentage
    - dark_spots_percentage
    - puffy_eyes_percentage
    - wrinkles_percentage
    - bbox_x
    - bbox_y
    - bbox_width
    - bbox_height
    
    Args:
        original_image (ndarray): Original input image (BGR)
        results (list): List of detection results
        class_names (list): Names of classes
        
    Returns:
        tuple: (csv_bytes, filename) or (None, None) if generation fails
    """
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"DermalScan_{timestamp}"
        filename = f"Skin_Analysis_Report_{timestamp}.csv"
        
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        
        # Header row with Infosys-compliant column names
        writer.writerow([
            'image_name',
            'face_id',
            'clear_skin_percentage',
            'dark_spots_percentage',
            'puffy_eyes_percentage',
            'wrinkles_percentage',
            'bbox_x',
            'bbox_y',
            'bbox_width',
            'bbox_height',
            'top_prediction',
            'confidence_percentage'
        ])
        
        # Data rows - one row per face
        for person_idx, result in enumerate(results, 1):
            x, y, w, h = result['box']
            predictions = result['all_predictions']
            
            row = [
                image_name,
                f'Face_{person_idx}',
                f"{predictions[0]:.2f}",  # Clear Skin
                f"{predictions[1]:.2f}",  # Dark Spots
                f"{predictions[2]:.2f}",  # Puffy Eyes
                f"{predictions[3]:.2f}",  # Wrinkles
                int(x),
                int(y),
                int(w),
                int(h),
                result['label'],
                f"{result['confidence']:.2f}"
            ]
            writer.writerow(row)
        
        # Convert to bytes
        csv_content = csv_buffer.getvalue()
        csv_bytes = io.BytesIO(csv_content.encode('utf-8'))
        csv_bytes.seek(0)
        
        logger.info(f"✓ CSV report generated successfully: {filename}")
        return csv_bytes, filename
        
    except Exception as e:
        logger.error(f"❌ Error generating CSV: {str(e)}", exc_info=True)
        return None, None


# ==================== JSON EXPORT ====================

def generate_json_report(original_image, results, class_names=CLASS_NAMES):
    """
    Generate JSON report for skin analysis results.
    
    Args:
        original_image (ndarray): Original input image (BGR)
        results (list): List of detection results
        class_names (list): Names of classes
        
    Returns:
        tuple: (json_bytes, filename) or (None, None) if generation fails
    """
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Skin_Analysis_Report_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'report_generated': datetime.now().isoformat(),
                'total_faces_detected': len(results),
                'model': 'EfficientNetB0',
                'accuracy': '90%'
            },
            'results': []
        }
        
        for person_idx, result in enumerate(results, 1):
            x, y, w, h = result['box']
            person_data = {
                'person_id': person_idx,
                'bounding_box': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                'top_prediction': {
                    'label': result['label'],
                    'confidence_percent': round(result['confidence'], 2)
                },
                'all_predictions': {}
            }
            
            for class_name, confidence in zip(class_names, result['all_predictions']):
                person_data['all_predictions'][class_name] = round(confidence, 2)
            
            report_data['results'].append(person_data)
        
        # Convert to JSON
        json_content = json.dumps(report_data, indent=2)
        json_bytes = io.BytesIO(json_content.encode('utf-8'))
        json_bytes.seek(0)
        
        logger.info(f"✓ JSON report generated successfully: {filename}")
        return json_bytes, filename
        
    except Exception as e:
        logger.error(f"❌ Error generating JSON: {str(e)}", exc_info=True)
        return None, None


# ==================== ANNOTATED IMAGE EXPORT ====================

def export_annotated_image(original_image, results):
    """
    Export high-quality annotated image with LARGE, HIGHLY VISIBLE text and bounding boxes.
    
    Requirements:
    - Keep original resolution (no downsampling)
    - Use LARGE text sizes for maximum visibility
    - High contrast colors with solid backgrounds
    - Clear, bold labels that stand out
    - Professional quality output
    
    Args:
        original_image (ndarray): Original input image (BGR), unmodified
        results (list): List of detection results
        
    Returns:
        tuple: (image_bytes, filename) or (None, None) if export fails
    """
    
    try:
        import cv2
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DermalScan_Annotated_{timestamp}.png"
        
        # Create a copy at ORIGINAL RESOLUTION - NO RESIZING
        annotated = original_image.copy()
        
        # Define colors for HIGH CONTRAST (BGR format)
        box_color = (0, 255, 0)         # Bright GREEN box
        person_label_color = (0, 255, 0)  # Green background
        prediction_color = (0, 0, 255)  # Bright RED for prediction
        text_color = (255, 255, 255)    # White text for contrast
        
        # Process each detected face
        for person_idx, result in enumerate(results, 1):
            x, y, w, h = result['box']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # ===== DRAW THICK BOUNDING BOX =====
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                box_color,
                thickness=8,  # ULTRA THICK for maximum visibility
                lineType=cv2.LINE_AA
            )
            
            # ===== DRAW PERSON LABEL (VERY LARGE) =====
            person_label = f"Person {person_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            person_font_scale = 2.5  # EXTRA LARGE for exported image
            person_thickness = 4  # THICKER for clarity
            
            # Get text size
            text_size = cv2.getTextSize(person_label, font, person_font_scale, person_thickness)[0]
            text_width = text_size[0]
            text_height = text_size[1]
            
            # LABEL BACKGROUND - ABOVE THE BOX (LARGER PADDING)
            label_padding = 20
            label_bg_y_start = max(0, y - text_height - label_padding - 10)
            label_bg_y_end = y - 10
            label_bg_x_start = max(0, x - 10)
            label_bg_x_end = min(annotated.shape[1], x + text_width + label_padding)
            
            # Draw solid background rectangle for label
            cv2.rectangle(
                annotated,
                (label_bg_x_start, label_bg_y_start),
                (label_bg_x_end, label_bg_y_end),
                person_label_color,
                thickness=-1,  # FILLED
                lineType=cv2.LINE_AA
            )
            
            # Add border to label background
            cv2.rectangle(
                annotated,
                (label_bg_x_start, label_bg_y_start),
                (label_bg_x_end, label_bg_y_end),
                (255, 255, 255),  # White border
                thickness=4,  # Thicker border for emphasis
                lineType=cv2.LINE_AA
            )
            
            # Draw person label text
            text_x = x + 10
            text_y = max(text_height + 5, y - label_padding)
            cv2.putText(
                annotated,
                person_label,
                (text_x, text_y),
                font,
                person_font_scale,
                text_color,
                person_thickness,
                lineType=cv2.LINE_AA
            )
            
            # ===== DRAW PREDICTION TEXT (VERY LARGE) - BELOW BOX =====
            pred_text = f"{result['label']} ({result['confidence']:.1f}%)"
            pred_font_scale = 2.0  # EXTRA LARGE
            pred_thickness = 4  # THICKER for clarity
            
            # Get prediction text size
            pred_size = cv2.getTextSize(pred_text, font, pred_font_scale, pred_thickness)[0]
            
            # Position BELOW the box with adequate spacing
            pred_spacing = 30
            pred_y = y + h + pred_spacing
            
            # Ensure we don't go out of bounds
            if pred_y + pred_size[1] < annotated.shape[0]:
                # Background for prediction text
                pred_padding = 15
                pred_bg_y_start = pred_y - pred_size[1] - pred_padding
                pred_bg_y_end = pred_y + 10
                pred_bg_x_start = max(0, x - 10)
                pred_bg_x_end = min(annotated.shape[1], x + pred_size[0] + pred_padding)
                
                # Draw prediction background
                cv2.rectangle(
                    annotated,
                    (pred_bg_x_start, pred_bg_y_start),
                    (pred_bg_x_end, pred_bg_y_end),
                    prediction_color,
                    thickness=-1,  # FILLED
                    lineType=cv2.LINE_AA
                )
                
                # Add border to prediction background
                cv2.rectangle(
                    annotated,
                    (pred_bg_x_start, pred_bg_y_start),
                    (pred_bg_x_end, pred_bg_y_end),
                    (255, 255, 255),  # White border
                    thickness=4,  # Thicker border for emphasis
                    lineType=cv2.LINE_AA
                )
                
                # Draw prediction text
                cv2.putText(
                    annotated,
                    pred_text,
                    (x + 10, pred_y - 5),
                    font,
                    pred_font_scale,
                    text_color,
                    pred_thickness,
                    lineType=cv2.LINE_AA
                )
        
        # ===== ENCODE AS PNG (HIGH QUALITY) =====
        success, image_bytes = cv2.imencode('.png', annotated, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not success:
            logger.error("❌ Failed to encode annotated image")
            return None, None
        
        # Convert to BytesIO
        image_io = io.BytesIO(image_bytes.tobytes())
        image_io.seek(0)
        
        logger.info(f"✓ High-quality annotated image exported: {filename} (Original resolution, LARGE visible text)")
        return image_io, filename
        
    except Exception as e:
        logger.error(f"❌ Error exporting annotated image: {str(e)}", exc_info=True)
        return None, None


# ==================== MAIN REPORT GENERATOR ====================

def generate_report(original_image, results, format_type='pdf', class_names=CLASS_NAMES):
    """
    Generate report in specified format.
    
    Args:
        original_image (ndarray): Original input image (BGR)
        results (list): List of detection results from predict_skin_condition()
        format_type (str): 'pdf', 'csv', or 'json'
        class_names (list): Names of classes
        
    Returns:
        tuple: (file_bytes, filename) or (None, None) if generation fails
    """
    
    if not results:
        logger.warning("⚠️ No results available for report generation")
        return None, None
    
    if format_type.lower() == 'pdf':
        return generate_pdf_report(original_image, results, class_names)
    elif format_type.lower() == 'csv':
        return generate_csv_report(original_image, results, class_names)
    elif format_type.lower() == 'json':
        return generate_json_report(original_image, results, class_names)
    else:
        logger.error(f"❌ Unsupported format: {format_type}")
        return None, None
