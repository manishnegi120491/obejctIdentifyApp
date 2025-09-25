import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

def test_with_different_confidence(image_path, confidence_thresholds=[0.3, 0.5, 0.7, 0.8]):
    """Test the same image with different confidence thresholds"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = "person_detector_final.pth"
    if not os.path.exists(model_path):
        print("❌ No trained model found!")
        return
    
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model([img_tensor])
    
    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    print(f"🔍 Testing image: {os.path.basename(image_path)}")
    print(f"📐 Image size: {img.size}")
    print(f"👥 Total raw detections: {len(boxes)}")
    print("=" * 60)
    
    for threshold in confidence_thresholds:
        # Filter by confidence
        keep = scores >= threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        
        print(f"\n🎯 Confidence threshold: {threshold}")
        print(f"   ✅ Detections: {len(filtered_boxes)}")
        
        if len(filtered_boxes) > 0:
            print(f"   📊 Confidence range: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
            print(f"   📊 Average confidence: {filtered_scores.mean():.3f}")
            
            # Show details for each detection
            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0
                print(f"      Detection {i+1}: conf={score:.3f}, size={width:.0f}x{height:.0f}, ratio={aspect_ratio:.2f}")
        else:
            print("   ⚠️  No detections above threshold")

def main():
    # Test the image with the horse
    image_path = "image/000000012748.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image {image_path} not found!")
        return
    
    print("🔧 Confidence Threshold Analysis")
    print("This will help you find the best threshold to filter out false positives like horses")
    print("=" * 80)
    
    test_with_different_confidence(image_path)
    
    print("\n" + "=" * 80)
    print("💡 Recommendations:")
    print("- Lower threshold (0.3-0.4): More detections, but more false positives")
    print("- Medium threshold (0.5-0.6): Balanced approach")
    print("- Higher threshold (0.7+): Fewer detections, but more accurate")
    print("- For your horse image, threshold 0.5+ should filter out the horse")

if __name__ == "__main__":
    main()

