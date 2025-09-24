import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import os

def test_person_detection():
    """Test person detection on a sample image"""
    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    # Model paths to try
    model_paths = [
        "person_detector_final.pth",
        "person_detector.pth",
        "person_detector_epoch_10.pth"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Error: No trained model found!")
        print("Please train the model first using train_person.py")
        return
    
    print(f"Loading model from {model_path}")
    
    # Load model
    model = fasterrcnn_resnet50_fpn(weights=None)  # Don't load pretrained weights
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test on multiple images
    image_dir = "val2017"
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} not found!")
        return
    
    # Get sample images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
    
    if len(image_files) == 0:
        print(f"No images found in {image_dir}")
        return
    
    # Test on first 3 images
    test_images = image_files[:3]
    
    for i, img_name in enumerate(test_images):
        print(f"\n--- Testing Image {i+1}: {img_name} ---")
        
        img_path = os.path.join(image_dir, img_name)
        
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            transform = T.ToTensor()
            img_tensor = transform(img).to(device)
            
            # Make prediction
            with torch.no_grad():
                predictions = model([img_tensor])
            
            prediction = predictions[0]
            
            # Extract results
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            confidence_threshold = 0.5
            keep = scores >= confidence_threshold
            
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            print(f"Total detections: {len(boxes)}")
            print(f"High-confidence detections (≥{confidence_threshold}): {len(filtered_boxes)}")
            
            if len(filtered_boxes) > 0:
                print("Detection details:")
                for j, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
                    x1, y1, x2, y2 = box
                    print(f"  Person {j+1}: Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Confidence={score:.3f}")
            else:
                print("No persons detected with high confidence")
                
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    print("\n✅ Testing completed!")

if __name__ == "__main__":
    test_person_detection()
