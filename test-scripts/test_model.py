import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

class PersonDetectorTester:
    def __init__(self, model_path="person_detector_final.pth", confidence_threshold=0.5):
        """Initialize the person detector tester"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        
        print(f"Using device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            # Try alternative model paths
            alternative_paths = [
                "person_detector_epoch_10.pth",
                "person_detector_epoch_9.pth",
                "person_detector_epoch_8.pth"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    self.model_path = alt_path
                    break
            else:
                raise FileNotFoundError("No trained model found! Please train the model first.")
        
        print(f"Loading model from: {self.model_path}")
        
        # Initialize model
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def detect_persons(self, image_path):
        """Detect persons in a single image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            original_size = img.size
            
            # Transform image
            transform = T.ToTensor()
            img_tensor = transform(img).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                predictions = self.model([img_tensor])
            
            prediction = predictions[0]
            
            # Extract results
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            keep = scores >= self.confidence_threshold
            filtered_boxes = boxes[keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            return {
                'image': img,
                'original_size': original_size,
                'all_boxes': boxes,
                'all_scores': scores,
                'all_labels': labels,
                'filtered_boxes': filtered_boxes,
                'filtered_scores': filtered_scores,
                'filtered_labels': filtered_labels,
                'num_detections': len(filtered_boxes)
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def visualize_detections(self, result, save_path=None, show_plot=True):
        """Visualize detections on the image"""
        if result is None:
            return
        
        img = result['image'].copy()
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for i, (box, score) in enumerate(zip(result['filtered_boxes'], result['filtered_scores'])):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Draw label
            label_text = f"Person {i+1}: {score:.3f}"
            draw.text((x1, y1-25), label_text, fill="red", font=font)
        
        # Add summary text
        summary_text = f"Detected {result['num_detections']} persons (threshold: {self.confidence_threshold})"
        draw.text((10, 10), summary_text, fill="green", font=font)
        
        if save_path:
            img.save(save_path)
            print(f"Visualization saved to: {save_path}")
        
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Person Detection Results - {result['num_detections']} persons detected")
            plt.show()
        
        return img
    
    def test_single_image(self, image_path, visualize=True, save_visualization=False):
        """Test detection on a single image"""
        print(f"\n🔍 Testing image: {os.path.basename(image_path)}")
        print("-" * 50)
        
        result = self.detect_persons(image_path)
        if result is None:
            return
        
        # Print detailed results
        print(f"Image size: {result['original_size']}")
        print(f"Total detections: {len(result['all_boxes'])}")
        print(f"High-confidence detections (≥{self.confidence_threshold}): {result['num_detections']}")
        
        if result['num_detections'] > 0:
            print("\nDetection details:")
            for i, (box, score) in enumerate(zip(result['filtered_boxes'], result['filtered_scores'])):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                print(f"  Person {i+1}:")
                print(f"    Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                print(f"    Size: {width:.1f} x {height:.1f} pixels")
                print(f"    Confidence: {score:.3f}")
        else:
            print("No persons detected with high confidence")
        
        # Visualize if requested
        if visualize:
            save_path = None
            if save_visualization:
                save_path = f"detection_{os.path.basename(image_path)}.jpg"
            self.visualize_detections(result, save_path)
        
        return result
    
    def test_multiple_images(self, image_dir, num_images=5, visualize=True):
        """Test detection on multiple images"""
        if not os.path.exists(image_dir):
            print(f"Error: Image directory {image_dir} not found!")
            return
        
        # Get image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) == 0:
            print(f"No images found in {image_dir}")
            return
        
        print(f"Found {len(image_files)} images. Testing first {min(num_images, len(image_files))} images...")
        
        results = []
        for i, img_name in enumerate(image_files[:num_images]):
            img_path = os.path.join(image_dir, img_name)
            result = self.test_single_image(img_path, visualize=visualize)
            if result:
                results.append(result)
        
        # Print summary
        self.print_summary(results)
        return results
    
    def print_summary(self, results):
        """Print summary of all test results"""
        if not results:
            return
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        total_detections = sum(r['num_detections'] for r in results)
        avg_detections = total_detections / len(results)
        
        print(f"Images tested: {len(results)}")
        print(f"Total persons detected: {total_detections}")
        print(f"Average persons per image: {avg_detections:.2f}")
        
        # Confidence score statistics
        all_scores = []
        for result in results:
            all_scores.extend(result['filtered_scores'])
        
        if all_scores:
            print(f"Confidence score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
            print(f"Average confidence: {np.mean(all_scores):.3f}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Test trained person detection model")
    parser.add_argument("--model", default="person_detector_final.pth", help="Path to trained model")
    parser.add_argument("--image", help="Path to single image to test")
    parser.add_argument("--image_dir", default="val2017", help="Directory containing images to test")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to test")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--no_visualize", action="store_true", help="Don't show visualizations")
    parser.add_argument("--save_viz", action="store_true", help="Save visualization images")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PersonDetectorTester(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.image:
        # Test single image
        if os.path.exists(args.image):
            tester.test_single_image(
                args.image, 
                visualize=not args.no_visualize,
                save_visualization=args.save_viz
            )
        else:
            print(f"Error: Image file {args.image} not found!")
    else:
        # Test multiple images
        tester.test_multiple_images(
            args.image_dir,
            num_images=args.num_images,
            visualize=not args.no_visualize
        )

if __name__ == "__main__":
    main()

