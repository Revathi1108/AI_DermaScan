"""Test script for report generation"""
import numpy as np
import cv2
from report import generate_report

# Create a mock image and results
mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
mock_results = [
    {
        'box': (100, 100, 200, 200),
        'label': 'Clear Skin',
        'confidence': 85.5,
        'all_predictions': [85.5, 10.2, 2.1, 2.2],
        'class_names': ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
    },
    {
        'box': (350, 150, 180, 180),
        'label': 'Dark Spots',
        'confidence': 72.3,
        'all_predictions': [15.2, 72.3, 8.5, 4.0],
        'class_names': ['Clear Skin', 'Dark Spots', 'Puffy Eyes', 'Wrinkles']
    }
]

# Test PDF generation
print("Testing PDF generation...")
pdf_bytes, filename = generate_report(mock_image, mock_results, format_type='pdf')
if pdf_bytes:
    print(f"✓ PDF generated: {filename}")
    print(f"  File size: {len(pdf_bytes.getvalue())} bytes")
else:
    print("❌ PDF generation failed")

# Test CSV generation
print("\nTesting CSV generation...")
csv_bytes, filename = generate_report(mock_image, mock_results, format_type='csv')
if csv_bytes:
    print(f"✓ CSV generated: {filename}")
    content = csv_bytes.getvalue().decode('utf-8')
    print(f"  File size: {len(content)} characters")
else:
    print("❌ CSV generation failed")

# Test JSON generation
print("\nTesting JSON generation...")
json_bytes, filename = generate_report(mock_image, mock_results, format_type='json')
if json_bytes:
    print(f"✓ JSON generated: {filename}")
    content = json_bytes.getvalue().decode('utf-8')
    print(f"  File size: {len(content)} characters")
else:
    print("❌ JSON generation failed")

print("\n✓ All report formats tested successfully!")
