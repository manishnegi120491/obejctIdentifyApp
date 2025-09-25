import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def clean_person_detection():
    """Clean person detection showing only valid detections (green boxes only)"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    model_path = "person_detector_final.pth"
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        return
    
    print(f"📦 Loading model: {model_path}")
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Test on custom images with clean visualization
    image_dir = "image"
    if not os.path.exists(image_dir):
        print(f"❌ Image directory {image_dir} not found!")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🖼️  Testing on {len(image_files)} custom images with CLEAN visualization...")
    print("=" * 80)
    
    for i, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_name)
        print(f"\n📸 Image {i}: {img_name}")
        print("-" * 60)
        
        try:
            # Load and process image
            img = Image.open(img_path).convert("RGB")
            original_size = img.size
            print(f"   📐 Image size: {original_size[0]} x {original_size[1]} pixels")
            
            transform = T.ToTensor()
            img_tensor = transform(img).to(device)
            
            # Make prediction
            with torch.no_grad():
                predictions = model([img_tensor])
            
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Apply smart filtering
            filtered_results = apply_smart_filtering(boxes, scores, labels, original_size)
            
            print(f"   👥 Raw detections: {len(boxes)}")
            print(f"   ✅ Valid person detections: {len(filtered_results['valid_boxes'])}")
            print(f"   ❌ Filtered out (false positives): {len(filtered_results['filtered_boxes'])}")
            
            if len(filtered_results['valid_boxes']) > 0:
                print(f"\n   📊 Valid Person Detections:")
                for j, (box, score) in enumerate(zip(filtered_results['valid_boxes'], filtered_results['valid_scores'])):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = height / width if width > 0 else 0
                    print(f"      Person {j+1}:")
                    print(f"        🎯 Confidence: {score:.3f}")
                    print(f"        📦 Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    print(f"        📏 Size: {width:.1f} x {height:.1f} pixels")
                    print(f"        📐 Aspect ratio: {aspect_ratio:.2f}")
            else:
                print("   ⚠️  No valid person detections found")
            
            # Create clean visualization (only green boxes)
            create_clean_visualization(img, filtered_results, img_name)
                
        except Exception as e:
            print(f"   ❌ Error processing image: {str(e)}")

def apply_smart_filtering(boxes, scores, labels, image_size):
    """Apply smart filtering that specifically targets horses and other false positives"""
    
    # Moderate confidence threshold
    confidence_threshold = 0.5
    
    # Get image dimensions
    img_width, img_height = image_size
    
    valid_boxes = []
    valid_scores = []
    filtered_boxes = []
    filtered_scores = []
    filter_reasons = []
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / width if width > 0 else 0
        
        # Skip if confidence is too low
        if score < confidence_threshold:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Low confidence ({score:.3f} < {confidence_threshold})")
            continue
        
        # Calculate position metrics
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        left_ratio = x1 / img_width
        right_ratio = x2 / img_width
        top_ratio = y1 / img_height
        bottom_ratio = y2 / img_height
        
        # HORSE DETECTION FILTERS
        # Horses are typically on the left side of images and have wide heads
        is_left_side = left_ratio < 0.3  # Left 30% of image
        is_wide_detection = width > height * 0.6  # Width is more than 60% of height
        has_wide_head = width > height * 0.5  # Head area is wide relative to height
        
        # Check if this looks like a horse detection
        horse_likelihood = 0
        if is_left_side:
            horse_likelihood += 2
        if is_wide_detection:
            horse_likelihood += 2
        if has_wide_head:
            horse_likelihood += 1
        if aspect_ratio < 1.8:  # Not very tall
            horse_likelihood += 1
        
        # If it has 3+ horse characteristics, filter it out
        if horse_likelihood >= 3:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            reasons = []
            if is_left_side:
                reasons.append("left side")
            if is_wide_detection:
                reasons.append("wide shape")
            if has_wide_head:
                reasons.append("wide head")
            if aspect_ratio < 1.8:
                reasons.append("not tall enough")
            filter_reasons.append(f"Likely horse detection ({', '.join(reasons)})")
            continue
        
        # Additional human-specific checks
        # Humans should be reasonably tall
        if aspect_ratio < 1.2:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Too wide for human (aspect ratio: {aspect_ratio:.2f})")
            continue
        
        # Size checks - not too small or too large
        box_area = width * height
        image_area = img_width * img_height
        area_ratio = box_area / image_area
        
        if area_ratio < 0.01:  # Too small
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Too small relative to image ({area_ratio:.3f})")
            continue
        
        if area_ratio > 0.7:  # Too large
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Too large relative to image ({area_ratio:.3f})")
            continue
        
        # If it passes all filters, it's likely a valid person
        valid_boxes.append(box)
        valid_scores.append(score)
    
    return {
        'valid_boxes': np.array(valid_boxes) if valid_boxes else np.array([]),
        'valid_scores': np.array(valid_scores) if valid_scores else np.array([]),
        'filtered_boxes': np.array(filtered_boxes) if filtered_boxes else np.array([]),
        'filtered_scores': np.array(filtered_scores) if filtered_scores else np.array([]),
        'filter_reasons': filter_reasons
    }

def create_clean_visualization(img, filtered_results, img_name):
    """Create clean visualization showing only valid detections (green boxes only)"""
    try:
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw ONLY valid detections in green (no red boxes)
        for i, (box, score) in enumerate(zip(filtered_results['valid_boxes'], filtered_results['valid_scores'])):
            x1, y1, x2, y2 = box
            # Draw thick green rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
            # Draw label with green background
            label = f"Person {i+1}: {score:.3f}"
            # Get text size for background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Draw background rectangle for text
            draw.rectangle([x1, y1-25, x1+text_width+10, y1-5], fill="green")
            draw.text((x1+5, y1-22), label, fill="white", font=font)
        
        # Add clean summary
        valid_count = len(filtered_results['valid_boxes'])
        summary = f"Valid Person Detections: {valid_count}"
        # Draw background for summary
        bbox = draw.textbbox((0, 0), summary, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.rectangle([10, 10, 10+text_width+20, 10+text_height+10], fill="blue")
        draw.text((20, 15), summary, fill="white", font=font)
        
        # Save clean visualization
        output_name = f"clean_detection_{img_name}"
        vis_img.save(output_name)
        print(f"   💾 Clean visualization saved: {output_name}")
        print(f"   🎯 Showing only {valid_count} valid person detection(s) in green")
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {str(e)}")

if __name__ == "__main__":
    clean_person_detection()

