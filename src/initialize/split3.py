import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dirs, target_base_dir, train_ratio=0.9):
    """
    Split images from source directories into train and validation sets
    
    Args:
        source_dirs: List of source directories (processed1, processed2)
        target_base_dir: Base directory for train and validation folders
        train_ratio: Ratio of images to use for training (default: 0.9)
    """
    
    # Class folder names
    class_folders = [
        "0Normal-processed",
        "1Doubtful-processed", 
        "2Mild-processed",
        "3Moderate-processed",
        "4Severe-processed"
    ]
    
    # Target class names (without -processed suffix)
    target_class_names = ["0", "1", "2", "3", "4"]
    
    for i, source_dir in enumerate(source_dirs, 1):
        print(f"Processing {source_dir}...")
        
        # Create target directories
        train_dir = os.path.join(target_base_dir, f"train{i}")
        val_dir = os.path.join(target_base_dir, f"validation{i}")
        
        for class_name in target_class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Process each class
        for class_folder, target_class in zip(class_folders, target_class_names):
            source_class_dir = os.path.join(source_dir, class_folder)
            
            if not os.path.exists(source_class_dir):
                print(f"Warning: {source_class_dir} does not exist, skipping...")
                continue
            
            # Get all image files
            image_files = []
            for file in os.listdir(source_class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(file)
            
            if not image_files:
                print(f"Warning: No images found in {source_class_dir}")
                continue
            
            # Shuffle and split
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_ratio)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            print(f"{class_folder}: {len(train_files)} train, {len(val_files)} validation")
            
            # Copy files to train directory
            for file in train_files:
                src_path = os.path.join(source_class_dir, file)
                dst_path = os.path.join(train_dir, target_class, file)
                shutil.copy2(src_path, dst_path)
            
            # Copy files to validation directory
            for file in val_files:
                src_path = os.path.join(source_class_dir, file)
                dst_path = os.path.join(val_dir, target_class, file)
                shutil.copy2(src_path, dst_path)

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    base_dir = "Data/KNEE-init-images"
    processed1_dir = os.path.join(base_dir, "processed1")
    processed2_dir = os.path.join(base_dir, "processed2")
    
    # Check if source directories exist
    if not os.path.exists(processed1_dir):
        print(f"Error: {processed1_dir} does not exist!")
        return
    if not os.path.exists(processed2_dir):
        print(f"Error: {processed2_dir} does not exist!")
        return
    
    # Create the splits
    print("Creating dataset splits...")
    split_dataset([processed1_dir, processed2_dir], base_dir)
    
    print("\nDataset splitting completed!")
    print("Created folders:")
    print("- train1/ with subfolders 0, 1, 2, 3, 4 (90% of processed1)")
    print("- validation1/ with subfolders 0, 1, 2, 3, 4 (10% of processed1)")
    print("- train2/ with subfolders 0, 1, 2, 3, 4 (90% of processed2)")
    print("- validation2/ with subfolders 0, 1, 2, 3, 4 (10% of processed2)")

if __name__ == "__main__":
    main()