import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

def load_trained_model(model_path, device):
    """Load the trained person detection model"""
    model = fasterrcnn_resnet50_fpn(weights=None)  # Don't load pretrained weights
    
    # Modify classifier for person detection (2 classes: background + person)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def visualize_predictions(image_path, model, device, confidence_threshold=0.5):
    """Visualize person detection results on an image"""
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    img_tensor = transform(img).to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model([img_tensor])
    
    prediction = predictions[0]
    
    # Filter predictions by confidence threshold
    scores = prediction['scores'].cpu().numpy()
    boxes = prediction['boxes'].cpu().numpy()
    
    # Keep only high-confidence detections
    keep = scores >= confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    
    # Create visualization
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw bounding boxes
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence score text
        ax.text(x1, y1-5, f'Person: {score:.2f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'Person Detection Results (Confidence ≥ {confidence_threshold})')
    ax.axis('off')
    
    return fig, len(boxes)

def test_model_on_images(model_path, image_dir, num_images=5, confidence_threshold=0.5):
    """Test the trained model on multiple images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_trained_model(model_path, device)
    print("Model loaded successfully!")
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"No image files found in {image_dir}")
        return
    
    # Test on selected images
    test_images = image_files[:num_images]
    
    total_detections = 0
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(image_dir, img_name)
        print(f"\nTesting image {i+1}/{len(test_images)}: {img_name}")
        
        try:
            # Visualize predictions
            fig, num_detections = visualize_predictions(img_path, model, device, confidence_threshold)
            total_detections += num_detections
            
            print(f"Detected {num_detections} persons")
            
            # Save visualization
            output_path = f"detection_result_{i+1}_{img_name}.png"
            fig.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {output_path}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
    
    print(f"\nValidation completed!")
    print(f"Total detections across {len(test_images)} images: {total_detections}")
    print(f"Average detections per image: {total_detections/len(test_images):.2f}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "person_detector_final.pth"  # Update this path if needed
    IMAGE_DIR = "val2017"
    CONFIDENCE_THRESHOLD = 0.5
    NUM_TEST_IMAGES = 5
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        print("Please train the model first using train_person.py")
        exit(1)
    
    # Run validation
    test_model_on_images(MODEL_PATH, IMAGE_DIR, NUM_TEST_IMAGES, CONFIDENCE_THRESHOLD)
