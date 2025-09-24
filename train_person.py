import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from dataset import CocoPersonDataset
import os
from tqdm import tqdm

def main():
    """Main training function"""
    # Paths - Updated to match current directory structure
    root = r"val2017"
    annFile = r"annotations\instances_val2017_person.json"

    # Check if files exist
    if not os.path.exists(root):
        print(f"Error: Image directory {root} not found!")
        return
    if not os.path.exists(annFile):
        print(f"Error: Annotation file {annFile} not found!")
        return

    # Data transforms with augmentation - only for images
    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    # Dataset
    dataset = CocoPersonDataset(root, annFile, transforms=transform)
    print(f"Dataset loaded with {len(dataset)} images")

    # Create data loader - Set num_workers=0 for Windows compatibility
    data_loader = DataLoader(
        dataset, 
        batch_size=2,  # Reduced for CPU training
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Modify the classifier for person detection (2 classes: background + person)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # Move model to device
    model.to(device)

    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training parameters
    num_epochs = 10
    print_every = 50

    # Training loop
    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += losses.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{losses.item():.4f}',
                'Avg Loss': f'{epoch_loss/num_batches:.4f}'
            })
            
            # Print detailed loss every print_every batches
            if (batch_idx + 1) % print_every == 0:
                print(f'\nEpoch {epoch+1}, Batch {batch_idx+1}:')
                for key, value in loss_dict.items():
                    print(f'  {key}: {value.item():.4f}')
                print(f'  Total Loss: {losses.item():.4f}')
        
        # Update learning rate
        lr_scheduler.step()
        
        # Print epoch summary
        avg_loss = epoch_loss / num_batches
        print(f'\nEpoch {epoch+1} completed!')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 50)
        
        # Save model checkpoint
        checkpoint_path = f'person_detector_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path}')

    # Save final model
    final_model_path = 'person_detector_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'\nTraining completed! Final model saved to {final_model_path}')

    # Test the trained model
    print("\nTesting the trained model...")
    model.eval()

    # Load a sample image for testing
    sample_images = os.listdir(root)[:3]  # Take first 3 images
    for img_name in sample_images:
        if img_name.endswith('.jpg'):
            img_path = os.path.join(root, img_name)
            
            # Load and preprocess image
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = T.ToTensor()(img).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model([img_tensor])
            
            print(f"\nImage: {img_name}")
            print(f"Detected {len(prediction[0]['boxes'])} persons")
            if len(prediction[0]['boxes']) > 0:
                print(f"Confidence scores: {prediction[0]['scores'].cpu().numpy()}")
            
    print("\n🎉 Training and testing completed successfully!")

if __name__ == '__main__':
    main()
