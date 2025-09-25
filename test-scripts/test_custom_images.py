import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os

def test_custom_images():
    """Test the trained model on custom images in the 'image' folder"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Find the best available model
    model_candidates = [
        "person_detector_final.pth",
        "person_detector_epoch_10.pth",
        "person_detector_epoch_9.pth",
        "person_detector_epoch_8.pth"
    ]
    
    model_path = None
    for candidate in model_candidates:
        if os.path.exists(candidate):
            model_path = candidate
            break
    
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print(f"📦 Loading model: {model_path}")
    
    # Load model
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Test on custom images
    image_dir = "image"
    if not os.path.exists(image_dir):
        print(f"❌ Image directory {image_dir} not found!")
        return
    
    # Get all images in the folder
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🖼️  Testing on {len(image_files)} custom images...")
    print("=" * 70)
    
    total_detections = 0
    confidence_threshold = 0.5
    
    for i, img_name in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, img_name)
        print(f"\n📸 Image {i}: {img_name}")
        print("-" * 50)
        
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
            
            # Filter by confidence
            high_conf = scores >= confidence_threshold
            filtered_boxes = boxes[high_conf]
            filtered_scores = scores[high_conf]
            filtered_labels = labels[high_conf]
            
            num_detections = len(filtered_boxes)
            total_detections += num_detections
            
            print(f"   👥 Total detections: {len(boxes)}")
            print(f"   ✅ High-confidence detections (≥{confidence_threshold}): {num_detections}")
            
            if num_detections > 0:
                print(f"\n   📊 Detection Details:")
                for j, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    print(f"      Person {j+1}:")
                    print(f"        🎯 Confidence: {score:.3f}")
                    print(f"        📦 Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                    print(f"        📏 Size: {width:.1f} x {height:.1f} pixels")
                    print(f"        📍 Center: ({x1+width/2:.1f}, {y1+height/2:.1f})")
            else:
                print("   ⚠️  No high-confidence detections")
                
                # Show low-confidence detections if any
                low_conf = scores < confidence_threshold
                if low_conf.any():
                    low_conf_boxes = boxes[low_conf]
                    low_conf_scores = scores[low_conf]
                    print(f"   ℹ️  Low-confidence detections: {len(low_conf_boxes)}")
                    for j, (box, score) in enumerate(zip(low_conf_boxes, low_conf_scores)):
                        print(f"      Detection {j+1}: confidence = {score:.3f}")
            
            # Create visualization
            create_visualization(img, filtered_boxes, filtered_scores, img_name)
                
        except Exception as e:
            print(f"   ❌ Error processing image: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 CUSTOM IMAGES TEST SUMMARY")
    print("=" * 70)
    print(f"Images tested: {len(image_files)}")
    print(f"Total persons detected: {total_detections}")
    print(f"Average per image: {total_detections/len(image_files):.1f}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("✅ Custom images test completed!")

def create_visualization(img, boxes, scores, img_name):
    """Create a simple visualization with bounding boxes"""
    try:
        # Create a copy for drawing
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw bounding boxes
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label = f"Person {i+1}: {score:.3f}"
            draw.text((x1, y1-20), label, fill="red", font=font)
        
        # Add summary
        summary = f"Detected {len(boxes)} persons"
        draw.text((10, 10), summary, fill="green", font=font)
        
        # Save visualization
        output_name = f"detection_{img_name}"
        vis_img.save(output_name)
        print(f"   💾 Visualization saved: {output_name}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {str(e)}")

if __name__ == "__main__":
    test_custom_images()

