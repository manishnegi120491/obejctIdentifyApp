import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def improved_person_detection():
    """Improved person detection with better filtering and validation"""
    
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
    
    # Test on custom images with improved filtering
    image_dir = "image"
    if not os.path.exists(image_dir):
        print(f"❌ Image directory {image_dir} not found!")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🖼️  Testing on {len(image_files)} custom images with improved filtering...")
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
            
            # Apply improved filtering
            filtered_results = apply_improved_filtering(boxes, scores, labels, original_size)
            
            print(f"   👥 Raw detections: {len(boxes)}")
            print(f"   ✅ Valid person detections: {len(filtered_results['valid_boxes'])}")
            print(f"   ❌ Filtered out (likely false positives): {len(filtered_results['filtered_boxes'])}")
            
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
            
            if len(filtered_results['filtered_boxes']) > 0:
                print(f"\n   ⚠️  Filtered Detections (likely false positives):")
                for j, (box, score, reason) in enumerate(zip(filtered_results['filtered_boxes'], 
                                                           filtered_results['filtered_scores'],
                                                           filtered_results['filter_reasons'])):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = height / width if width > 0 else 0
                    print(f"      Detection {j+1}:")
                    print(f"        🎯 Confidence: {score:.3f}")
                    print(f"        📦 Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    print(f"        📏 Size: {width:.1f} x {height:.1f} pixels")
                    print(f"        📐 Aspect ratio: {aspect_ratio:.2f}")
                    print(f"        ❌ Filtered because: {reason}")
            
            # Create improved visualization
            create_improved_visualization(img, filtered_results, img_name)
                
        except Exception as e:
            print(f"   ❌ Error processing image: {str(e)}")

def apply_improved_filtering(boxes, scores, labels, image_size):
    """Apply improved filtering to reduce false positives like horses"""
    
    # Basic confidence threshold
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
        
        # Filter based on aspect ratio (humans are typically taller than wide)
        if aspect_ratio < 1.2:  # Too wide relative to height
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Unusual aspect ratio ({aspect_ratio:.2f} - too wide)")
            continue
        
        # Filter based on size relative to image
        box_area = width * height
        image_area = img_width * img_height
        area_ratio = box_area / image_area
        
        if area_ratio < 0.01:  # Too small relative to image
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Too small relative to image ({area_ratio:.3f})")
            continue
        
        if area_ratio > 0.8:  # Too large relative to image
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Too large relative to image ({area_ratio:.3f})")
            continue
        
        # Filter based on position (humans are typically not at extreme edges)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if detection is too close to edges
        edge_threshold = 0.1  # 10% from edges
        if (center_x < img_width * edge_threshold or 
            center_x > img_width * (1 - edge_threshold) or
            center_y < img_height * edge_threshold or 
            center_y > img_height * (1 - edge_threshold)):
            # This might be a partial detection, but let's be more lenient
            pass
        
        # Additional check: if the detection is very wide and not very tall, it's likely not a person
        if width > height * 0.8 and aspect_ratio < 1.5:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            filter_reasons.append(f"Shape suggests non-human (width={width:.1f}, height={height:.1f})")
            continue
        
        # If it passes all filters, it's likely a valid person detection
        valid_boxes.append(box)
        valid_scores.append(score)
    
    return {
        'valid_boxes': np.array(valid_boxes) if valid_boxes else np.array([]),
        'valid_scores': np.array(valid_scores) if valid_scores else np.array([]),
        'filtered_boxes': np.array(filtered_boxes) if filtered_boxes else np.array([]),
        'filtered_scores': np.array(filtered_scores) if filtered_scores else np.array([]),
        'filter_reasons': filter_reasons
    }

def create_improved_visualization(img, filtered_results, img_name):
    """Create visualization with different colors for valid and filtered detections"""
    try:
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw valid detections in green
        for i, (box, score) in enumerate(zip(filtered_results['valid_boxes'], filtered_results['valid_scores'])):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            label = f"Person {i+1}: {score:.3f}"
            draw.text((x1, y1-20), label, fill="green", font=font)
        
        # Draw filtered detections in red (with strikethrough effect)
        for i, (box, score) in enumerate(zip(filtered_results['filtered_boxes'], filtered_results['filtered_scores'])):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label = f"Filtered {i+1}: {score:.3f}"
            draw.text((x1, y1-20), label, fill="red", font=font)
        
        # Add summary
        valid_count = len(filtered_results['valid_boxes'])
        filtered_count = len(filtered_results['filtered_boxes'])
        summary = f"Valid: {valid_count} | Filtered: {filtered_count}"
        draw.text((10, 10), summary, fill="blue", font=font)
        
        # Save visualization
        output_name = f"improved_detection_{img_name}"
        vis_img.save(output_name)
        print(f"   💾 Improved visualization saved: {output_name}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {str(e)}")

if __name__ == "__main__":
    improved_person_detection()

