import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# --- AUTO-INSTALL ULTRALYTICS ---
try:
    import ultralytics
except ImportError:
    print("Ultralytics not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

from ultralytics import YOLO
from google.colab import drive

# ==========================================
# 0. SETUP & PATH CONFIGURATION
# ==========================================
def setup_environment():
    # Mount Google Drive
    drive.mount('/content/gdrive')

    # Define the Base Directory
    # Based on your prompt: '/content/gdrive/MyDrive/Colab Notebooks/project3_data'
    BASE_DIR = '/content/gdrive/MyDrive/Colab Notebooks/project3_data'

    # Check if directory exists
    if not os.path.exists(BASE_DIR):
        print(f"WARNING: Directory {BASE_DIR} not found. Please check your Drive paths.")
    else:
        print(f"Working directory set to: {BASE_DIR}")

    return BASE_DIR

# ==========================================
# 1. PART 1: OBJECT MASKING (OpenCV)
# ==========================================
def step1_object_masking(base_dir):
    print("\n--- Starting Step 1: Object Masking ---")

    # Construct path to the image
    # Assuming the motherboard_image.JPEG is inside the project3_data folder
    img_path = os.path.join(base_dir, 'motherboard_image.JPEG')

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 1. Read the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for matplotlib display

    # 2. Pre-processing (Blur to reduce noise)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Thresholding
    # PROBLEM: The previous threshold (60) was too low. The motherboard (dark grey) was being treated as background.
    # We increase it to catch the motherboard pixels (which are darker than the table but lighter than deep black).
    # We also use MORPH_CLOSE to fill in the bright components (gold heatsinks) so the board looks like one solid block.
    _, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to fill holes (components) and remove noise
    kernel = np.ones((15, 15), np.uint8) # Large kernel to bridge gaps between components
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Optional: Open to remove small white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter Contours (Keep the largest one -> The Motherboard)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Create the final clean mask
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 6. Extract the PCB using bitwise_and
        result = cv2.bitwise_and(img, img, mask=final_mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # 7. Visualization (Required for Report)
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Generated Mask")
        plt.imshow(final_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Extracted PCB")
        plt.imshow(result_rgb)
        plt.axis('off')

        plt.show()

        # Save the result if needed
        output_path = os.path.join(base_dir, 'masked_motherboard_result.jpg')
        cv2.imwrite(output_path, result)
        print(f"Masked image saved to: {output_path}")

    else:
        print("No contours found. Check threshold values.")

# ==========================================
# 2. PART 2: YOLOv11 TRAINING
# ==========================================
def step2_yolo_training(base_dir):
    print("\n--- Starting Step 2: YOLOv11 Training ---")

    # Path to your data.yaml
    yaml_path = os.path.join(base_dir, 'data.yaml')

    # Load the model (Using YOLOv11 Nano as recommended)
    # Note: Ensure 'ultralytics' is installed via: !pip install ultralytics
    model = YOLO('yolo11n.pt')

    # HYPERPARAMETERS (Based on PDF requirements)
    # epochs: Must be below 200
    # imgsz: Minimum 900
    # batch: T4 GPU usually handles 16 well, lower to 8 if OutOfMemory errors occur
    EPOCHS = 100
    IMG_SIZE = 900
    BATCH_SIZE = 8
    PROJECT_PATH = os.path.join(base_dir, 'runs/detect')
    NAME = 'pcb_component_model'

    print(f"Training with: Epochs={EPOCHS}, ImgSz={IMG_SIZE}, Batch={BATCH_SIZE}")

    # Start Training
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_PATH,
        name=NAME,
        exist_ok=True, # Overwrite if exists
        plots=True,     # Automatically generates confusion matrix & curves
        workers = 2     # LIMIT RAM USAGE CAUSE IT CRASHES
    )

    print("Training Complete.")
    return model, os.path.join(PROJECT_PATH, NAME)

# ==========================================
# 3. PART 3: EVALUATION
# ==========================================
def step3_evaluation(model, base_dir, save_dir):
    print("\n--- Starting Step 3: Evaluation ---")

    # Path to evaluation images
    eval_folder = os.path.join(base_dir, 'evaluation')

    if not os.path.exists(eval_folder):
        print(f"Evaluation folder not found at {eval_folder}")
        return

    # Get list of images in evaluation folder
    image_files = [f for f in os.listdir(eval_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_files)} images for evaluation.")

    # Run Inference
    # We use the same imgsz as training (900)
    results = model.predict(
        source=eval_folder,
        imgsz=900,
        conf=0.25,      # Confidence threshold (adjust if missing components)
        save=True,      # Save the images with bounding boxes
        project=save_dir,
        name='evaluation_results'
    )

    # Display Results (Optional in Notebook)
    for result in results:
        print(f"Processed: {result.path}")
        # You can plot here if you want to see immediate output
        # im_array = result.plot()  # plot a BGR numpy array of predictions
        # plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        # plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    base_directory = setup_environment()

    # 2. Run Masking
    step1_object_masking(base_directory)

    # 3. Run Training
    # NOTE: This takes a long time.
    trained_model, run_path = step2_yolo_training(base_directory)

    # 4. Run Evaluation
    step3_evaluation(trained_model, base_directory, run_path)