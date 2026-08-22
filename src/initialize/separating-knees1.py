import os
import cv2
import numpy as np
from pathlib import Path
import shutil
############################ seperating knees
def process_knee_images(input_folder):
    """
    Process knee X-ray images: split wide images containing both legs into separate left/right images
    """
    # Create output folder path
    input_path = Path(input_folder)
    output_folder = input_path.parent / f"{input_path.name}_semi-processed"
    
    # Create output directory
    output_folder.mkdir(exist_ok=True)
    print(f"Created output folder: {output_folder}")
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dcm'}
    
    # Counters for statistics
    total_images = 0
    split_images = 0
    copied_images = 0
    
    # Process each file in the input folder
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            total_images += 1
            print(f"Processing: {file_path.name}")
            
            try:
                # Read image
                if file_path.suffix.lower() == '.dcm':
                    # For DICOM files
                    try:
                        import pydicom
                        ds = pydicom.dcmread(str(file_path))
                        img = ds.pixel_array
                        # Convert to 8-bit if necessary
                        if img.dtype != np.uint8:
                            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    except ImportError:
                        print("pydicom not installed, skipping DICOM file")
                        continue
                else:
                    # For regular image files
                    img = cv2.imread(str(file_path))
                
                if img is None:
                    print(f"  Could not read image: {file_path.name}")
                    continue
                
                # Get image dimensions
                height, width = img.shape[:2]
                
                # Check if image needs to be split
                if width > 2.5 * height:
                    print(f"  Splitting image (width: {width}, height: {height})")
                    split_images += 1
                    
                    # Calculate middle point
                    middle_x = width // 2
                    
                    # Extract left and right halves
                    # Note: In medical imaging, the left side of the image is typically the patient's right
                    right_leg = img[:, :middle_x]  # Left side of image = patient's RIGHT leg
                    left_leg = img[:, middle_x:]   # Right side of image = patient's LEFT leg
                    
                    # Prepare filename parts
                    file_stem = file_path.stem
                    file_extension = file_path.suffix
                    
                    # Save right leg (left side of image)
                    right_filename = f"{file_stem}_R{file_extension}"
                    right_output_path = output_folder / right_filename
                    cv2.imwrite(str(right_output_path), right_leg)
                    print(f"  Saved right leg: {right_filename}")
                    
                    # Save left leg (right side of image)
                    left_filename = f"{file_stem}_L{file_extension}"
                    left_output_path = output_folder / left_filename
                    cv2.imwrite(str(left_output_path), left_leg)
                    print(f"  Saved left leg: {left_filename}")
                    
                else:
                    # Copy original image if no splitting needed
                    copied_images += 1
                    output_path = output_folder / file_path.name
                    if file_path.suffix.lower() == '.dcm':
                        # For DICOM, we need special handling
                        try:
                            import pydicom
                            ds.save_as(str(output_path))
                        except:
                            # Fallback: convert to PNG if DICOM copy fails
                            output_path = output_folder / f"{file_path.stem}.png"
                            cv2.imwrite(str(output_path), img)
                    else:
                        shutil.copy2(file_path, output_path)
                    print(f"  Copied single-leg image: {file_path.name}")
                    
            except Exception as e:
                print(f"  Error processing {file_path.name}: {str(e)}")
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total images processed: {total_images}")
    print(f"Images split into left/right: {split_images}")
    print(f"Single-leg images copied: {copied_images}")
    print(f"Output folder: {output_folder}")
    
    return output_folder

def preview_processing(input_folder, num_samples=5):
    """
    Preview what the processing will do without actually processing images
    """
    print("=== PREVIEW MODE ===")
    input_path = Path(input_folder)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dcm'}
    sample_count = 0
    
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions and sample_count < num_samples:
            try:
                if file_path.suffix.lower() == '.dcm':
                    import pydicom
                    ds = pydicom.dcmread(str(file_path))
                    img = ds.pixel_array
                    height, width = img.shape[:2]
                else:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        height, width = img.shape[:2]
                    else:
                        continue
                
                print(f"Image: {file_path.name}")
                print(f"  Dimensions: {width} x {height} (width x height)")
                print(f"  Width/Height ratio: {width/height:.2f}")
                
                if width > 2.5 * height:
                    print(f"  → WILL BE SPLIT into left and right images")
                    middle_x = width // 2
                    print(f"  → Split point: {middle_x} pixels")
                    print(f"  → Output files: {file_path.stem}_L{file_path.suffix}, {file_path.stem}_R{file_path.suffix}")
                else:
                    print(f"  → WILL BE COPIED as single-leg image")
                
                print()
                sample_count += 1
                
            except Exception as e:
                print(f"  Error previewing {file_path.name}: {str(e)}")
    
    print(f"Previewed {sample_count} sample images")
    print("Run process_knee_images() to actually process all images")

# Main execution
if __name__ == "__main__":
    # Example usage
    input_folder = r"Data\KNEE-init-images\MedicalExpert-II\4Severe"  # Replace with your folder path
    
    # First, preview what will happen
    print("Would you like to preview the processing first? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        preview_processing(input_folder)
        print("\nProceed with actual processing? (y/n)")
        if input().strip().lower() == 'y':
            output_folder = process_knee_images(input_folder)
        else:
            print("Processing cancelled.")
    else:
        # Process directly
        output_folder = process_knee_images(input_folder)