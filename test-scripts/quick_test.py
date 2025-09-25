import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os
import glob

def quick_test():
    """Quick test of the trained model on sample images"""
    
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
        print("Available files:")
        for f in os.listdir("."):
            if f.endswith(".pth"):
                print(f"  - {f}")
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
    
    # Test on sample images
    image_dir = "val2017"
    if not os.path.exists(image_dir):
        print(f"❌ Image directory {image_dir} not found!")
        return
    
    # Get sample images
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))[:5]  # Test first 5 images
    
    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return
    
    print(f"🖼️  Testing on {len(image_files)} images...")
    print("=" * 60)
    
    total_detections = 0
    confidence_threshold = 0.5
    
    for i, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        print(f"\n📸 Image {i}: {img_name}")
        
        try:
            # Load and process image
            img = Image.open(img_path).convert("RGB")
            transform = T.ToTensor()
            img_tensor = transform(img).to(device)
            
            # Make prediction
            with torch.no_grad():
                predictions = model([img_tensor])
            
            prediction = predictions[0]
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            # Filter by confidence
            high_conf = scores >= confidence_threshold
            filtered_boxes = boxes[high_conf]
            filtered_scores = scores[high_conf]
            
            num_detections = len(filtered_boxes)
            total_detections += num_detections
            
            print(f"   👥 Detected: {num_detections} persons")
            
            if num_detections > 0:
                print(f"   📊 Confidence scores: {filtered_scores}")
                print(f"   📏 Confidence range: {filtered_scores.min():.3f} - {filtered_scores.max():.3f}")
            else:
                print("   ⚠️  No high-confidence detections")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Images tested: {len(image_files)}")
    print(f"Total persons detected: {total_detections}")
    print(f"Average per image: {total_detections/len(image_files):.1f}")
    print(f"Confidence threshold: {confidence_threshold}")
    print("✅ Quick test completed!")

if __name__ == "__main__":
    quick_test()

