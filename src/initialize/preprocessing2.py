import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

def make_image_square(image):
    """
    Ensure the image is square by extending the shorter dimension appropriately
    """
    height, width = image.shape[:2]
    
    # If already square, return as is
    if width == height:
        return image
    
    # If width > height: extend height by adding borders
    elif width > height:
        difference = width - height
        top_pad = difference // 2
        bottom_pad = difference - top_pad
        
        # Use border replication to extend the image
        squared_image = cv2.copyMakeBorder(
            image, 
            top_pad, 
            bottom_pad, 
            0, 0, 
            cv2.BORDER_REPLICATE
        )
        return squared_image
    
    # If height > width: extend width
    else:  # height > width
        difference = height - width
        left_pad = difference // 2
        right_pad = difference - left_pad
        
        # Use border replication to extend the image
        squared_image = cv2.copyMakeBorder(
            image, 
            0, 0, 
            left_pad, 
            right_pad, 
            cv2.BORDER_REPLICATE
        )
        return squared_image

def show_processing_steps_separate(image_path):
    """
    Display all processing steps in separate windows, one after another
    """
    # Read original image
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create duplicate
    duplicate = original.copy()
    
    # Convert duplicate to grayscale
    gray = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
    
    # Find threshold from histogram (NO BLUR)
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    total_pixels = gray.size
    threshold_value = 50  # default
    
    for i in range(len(cdf)):
        if cdf[i] >= total_pixels * 0.6:
            threshold_value = i
            break
    
    # Convert to binary
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    # Closing with circle radius 7
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Opening with circle radius 7  
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Find boundaries on horizontal axis
    height, width = opened.shape
    column_sums = np.sum(opened, axis=0)
    
    # Find left boundary
    left_edge = 0
    for i in range(width):
        if column_sums[i] > height * 10:
            left_edge = max(0, i - 5)
            break
    
    # Find right boundary
    right_edge = width - 1
    for i in range(width-1, -1, -1):
        if column_sums[i] > height * 10:
            right_edge = min(width-1, i + 5)
            break
    
    # Crop original image based on binary boundaries
    cropped = original[:, left_edge:right_edge]
    
    # Make the image square
    square_image = make_image_square(cropped)
    
    # Resize to 224x224
    final_image = cv2.resize(square_image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # WINDOW 1: First 4 images (Original, Grayscale, Binary, After Closing)
    print("\n--- Showing Window 1: Initial Processing Steps ---")
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f'Window 1: Initial Processing Steps - {image_path.name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes1[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes1[0, 0].set_title('1. Original Image', fontsize=12)
    axes1[0, 0].axis('off')
    
    # Grayscale
    axes1[0, 1].imshow(gray, cmap='gray')
    axes1[0, 1].set_title('2. Grayscale', fontsize=12)
    axes1[0, 1].axis('off')
    
    # Binary
    axes1[1, 0].imshow(binary, cmap='gray')
    axes1[1, 0].set_title(f'3. Binary (Threshold: {threshold_value})', fontsize=12)
    axes1[1, 0].axis('off')
    
    # After closing
    axes1[1, 1].imshow(closed, cmap='gray')
    axes1[1, 1].set_title('4. After Closing (radius 7)', fontsize=12)
    axes1[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(f'Window 2: Boundary Detection & Cropping - {image_path.name}', fontsize=16, fontweight='bold')
    
    # After opening with boundaries
    axes2[0, 0].imshow(opened, cmap='gray')
    axes2[0, 0].set_title('5. After Opening (radius 7)', fontsize=12)
    axes2[0, 0].axvline(x=left_edge, color='red', linestyle='--', linewidth=2, label=f'Left: {left_edge}')
    axes2[0, 0].axvline(x=right_edge, color='green', linestyle='--', linewidth=2, label=f'Right: {right_edge}')
    axes2[0, 0].legend(fontsize=9)
    axes2[0, 0].axis('off')
    
    # Column sums plot
    axes2[0, 1].plot(column_sums, 'b-', linewidth=2)
    axes2[0, 1].axvline(x=left_edge, color='red', linestyle='--', linewidth=2, label=f'Left: {left_edge}')
    axes2[0, 1].axvline(x=right_edge, color='green', linestyle='--', linewidth=2, label=f'Right: {right_edge}')
    axes2[0, 1].set_title('Column Sums (Binary Image)', fontsize=12)
    axes2[0, 1].set_xlabel('X Position', fontsize=10)
    axes2[0, 1].set_ylabel('Sum of White Pixels', fontsize=10)
    axes2[0, 1].legend(fontsize=9)
    axes2[0, 1].grid(True, alpha=0.3)
    
    # Cropped image (before squaring)
    axes2[1, 0].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axes2[1, 0].set_title(f'6. Cropped Image\n{cropped.shape[1]}x{cropped.shape[0]}', fontsize=12)
    axes2[1, 0].axis('off')
    
    # Square image (before resizing)
    axes2[1, 1].imshow(cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB))
    axes2[1, 1].set_title(f'7. Square Image\n{square_image.shape[1]}x{square_image.shape[0]}', fontsize=12)
    axes2[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
        
    # WINDOW 3: First (Original) and Last (Final 224x224) images
    print("\n--- Showing Window 3: Before & After Comparison ---")
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 6))
    fig3.suptitle(f'Window 3: Before & After Comparison - {image_path.name}', fontsize=16, fontweight='bold')
    
    # Original image
    axes3[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes3[0].set_title(f'1. Original Image\n{original.shape[1]}x{original.shape[0]}', fontsize=14)
    axes3[0].axis('off')
    
    # Final 224x224 image
    axes3[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    axes3[1].set_title('8. Final 224x224 Image', fontsize=14)
    axes3[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
        
    # WINDOW 4: Processing Information Summary
    print("\n--- Showing Window 4: Processing Information Summary ---")
    fig4 = plt.figure(figsize=(10, 8))
    fig4.suptitle(f'Processing Information Summary - {image_path.name}', fontsize=16, fontweight='bold')
    
    # Add processing information
    info_text = f'PROCESSING INFORMATION:\n\n'
    info_text += f'• Original: {original.shape[1]} x {original.shape[0]}\n'
    info_text += f'• Threshold: {threshold_value}\n'
    info_text += f'• Left boundary: {left_edge}\n'
    info_text += f'• Right boundary: {right_edge}\n'
    info_text += f'• ROI width: {right_edge - left_edge}\n'
    info_text += f'• Cropped: {cropped.shape[1]} x {cropped.shape[0]}\n'
    info_text += f'• Square: {square_image.shape[1]} x {square_image.shape[0]}\n'
    info_text += f'• Final: {final_image.shape[1]} x {final_image.shape[0]}\n\n'
    
    # Add square transformation info
    if cropped.shape[1] > cropped.shape[0]:
        info_text += f'• Square Transformation: Extended height using border replication\n'
    elif cropped.shape[1] < cropped.shape[0]:
        info_text += f'• Square Transformation: Extended width using border replication\n'
    else:
        info_text += f'• Square Transformation: Already square - no changes needed\n'
    
    info_text += f'• Resize: {square_image.shape[1]}x{square_image.shape[0]} → 224x224\n\n'
    
    # Add algorithm summary
    info_text += f'ALGORITHM STEPS:\n'
    info_text += f'1. Read image and create duplicate\n'
    info_text += f'2. Convert duplicate to grayscale\n'
    info_text += f'3. Find threshold from histogram (60%)\n'
    info_text += f'4. Convert to binary using threshold\n'
    info_text += f'5. Apply morphological closing (radius 7)\n'
    info_text += f'6. Apply morphological opening (radius 7)\n'
    info_text += f'7. Find boundaries using column sums\n'
    info_text += f'8. Crop original image\n'
    info_text += f'9. Make image square\n'
    info_text += f'10. Resize to 224x224'
    
    plt.text(0.1, 0.5, info_text, fontsize=12, va='center', linespacing=1.6, 
             bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print processing information to console
    print(f"\n=== PROCESSING INFORMATION ===")
    print(f"Original size: {original.shape[1]} x {original.shape[0]}")
    print(f"Threshold value: {threshold_value}")
    print(f"Detected boundaries - Left: {left_edge}, Right: {right_edge}")
    print(f"ROI width: {right_edge - left_edge}")
    print(f"Cropped size: {cropped.shape[1]} x {cropped.shape[0]}")
    print(f"Square size: {square_image.shape[1]} x {square_image.shape[0]}")
    print(f"Final size: {final_image.shape[1]} x {final_image.shape[0]}")
    
    # Print transformation type
    if cropped.shape[1] > cropped.shape[0]:
        print(f"Square transformation: Extended height using border replication")
    elif cropped.shape[1] < cropped.shape[0]:
        print(f"Square transformation: Extended width using border replication")
    else:
        print(f"Square transformation: Already square - no changes needed")
    print(f"Resize: {square_image.shape[1]}x{square_image.shape[0]} → 224x224")
    print("=" * 40)
    
    return final_image

def preview_random_samples(input_folder, num_samples=3):
    """
    Preview random samples from the dataset with all processing steps in separate windows
    """
    input_path = Path(input_folder)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # Get all valid image files
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in valid_extensions]
    
    if not image_files:
        print(f"No valid images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images in dataset")
    print(f"Showing {min(num_samples, len(image_files))} random samples...\n")
    
    # Select random samples
    selected_samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for i, sample_path in enumerate(selected_samples, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}/{len(selected_samples)}: {sample_path.name}")
        print(f"{'='*60}")
        
        final_image = show_processing_steps_separate(sample_path)
        
        if i < len(selected_samples):
            continue_input = input("\nPress Enter to see next sample, or 'q' to quit preview: ")
            if continue_input.lower() == 'q':
                break

def process_image_batch(image_path, output_path=None):
    """
    Process image for batch processing without visualization
    """
    # Read original image
    original = cv2.imread(str(image_path))
    if original is None:
        return None
    
    # Create duplicate
    duplicate = original.copy()
    
    # Convert duplicate to grayscale
    gray = cv2.cvtColor(duplicate, cv2.COLOR_BGR2GRAY)
    
    # Find threshold from histogram
    hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    total_pixels = gray.size
    threshold_value = 50
    
    for i in range(len(cdf)):
        if cdf[i] >= total_pixels * 0.6:
            threshold_value = i
            break
    
    # Convert to binary
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    
    # Find boundaries
    height, width = opened.shape
    column_sums = np.sum(opened, axis=0)
    
    left_edge = 0
    for i in range(width):
        if column_sums[i] > height * 10:
            left_edge = max(0, i - 5)
            break
    
    right_edge = width - 1
    for i in range(width-1, -1, -1):
        if column_sums[i] > height * 10:
            right_edge = min(width-1, i + 5)
            break
    
    # Crop original image
    cropped = original[:, left_edge:right_edge]
    
    # Make the image square
    square_image = make_image_square(cropped)
    
    # Resize to 224x224
    final_image = cv2.resize(square_image, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Save if output path provided
    if output_path is not None:
        cv2.imwrite(str(output_path), final_image)
    
    return final_image

def process_folder(input_folder, output_folder=None):
    """
    Process all images in a folder
    """
    input_path = Path(input_folder)
    
    # Create output folder
    if output_folder is None:
        output_path = input_path.parent / f"{input_path.name}_processed55"#########################################
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(exist_ok=True)
    
    # Process each image
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in valid_extensions]
    
    print(f"Processing {len(image_files)} images...")
    
    successful = 0
    for i, file_path in enumerate(image_files, 1):
        print(f"{i}/{len(image_files)}: Processing {file_path.name}")
        
        output_file = output_path / file_path.name
        result = process_image_batch(file_path, output_file)
        
        if result is not None:
            print(f"  ✓ Saved to: {output_file.name}")
            successful += 1
        else:
            print(f"  ✗ Failed to process")
    
    print(f"\nProcessing complete! {successful}/{len(image_files)} images processed successfully.")
    print(f"All images resized to 224x224")
    print(f"Output folder: {output_path}")
    return output_path

def show_algorithm_steps():
    """
    Display the enhanced algorithm steps
    """
    print("\n" + "="*70)
    print("ENHANCED ALGORITHM STEPS (NO BLUR + SQUARE + 224x224 RESIZE)")
    print("="*70)
    steps = [
        "1. Read image and create duplicate",
        "2. Convert duplicate to grayscale", 
        "3. Find threshold from histogram (60% of pixels below)",
        "4. Convert to binary using threshold",
        "5. Apply morphological CLOSING (circle radius 7)",
        "6. Apply morphological OPENING (circle radius 7)",
        "7. Find boundaries using column sums of binary image",
        "8. Crop original image using detected boundaries",
        "9. Ensure square output:",
        "   - If width > height: Extend height with border replication",
        "   - If width = height: Keep as is", 
        "   - If height > width: Extend width with border replication",
        "10. Resize square image to 224x224 using INTER_AREA interpolation"
    ]
    
    for step in steps:
        print(step)
    print("="*70)

# Main execution
if __name__ == "__main__":
    input_folder = r"KNEE_unimportant\z-other\Data\KNEE-init-images\semi-processed-1\0Normal_semi-processed"#replacable
    
    
    print("=" * 70)
    
    show_algorithm_steps()
    
    while True:
        print("\nChoose an option:")
        print("1. Preview random samples with all processing steps (separate windows)")
        print("2. Process entire dataset without preview")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            try:
                num_samples = int(input("How many random samples to preview? (default: 3): ") or "3")
            except ValueError:
                num_samples = 3
                
            preview_random_samples(input_folder, num_samples)
            
            # Ask if user wants to process all after preview
            proceed = input("\nDo you want to process ALL images now? (y/n): ").strip().lower()
            if proceed == 'y':
                output_folder = process_folder(input_folder)
                print(f"\nProcessing complete! Results saved to: {output_folder}")
                break
                
        elif choice == '2':
            output_folder = process_folder(input_folder)
            print(f"\nProcessing complete! Results saved to: {output_folder}")
            break
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

## C:/Users/HP/AppData/Local/Programs/Python/Python310/python.exe d:/AAA-projects-to-do/KNEE/src/initialize/preprocessing2.py