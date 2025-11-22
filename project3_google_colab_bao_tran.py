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
# --- USER CONFIGURATION (New) ---
# Set the desired mode for execution:
# 1. "FULL_RUN": Performs Masking, Training, and Evaluation.
# 2. "EVALUATION_ONLY": Skips training and loads weights from the specified path below.
MAIN_MODE = "EVALUATION_ONLY" 

# REQUIRED FOR EVALUATION_ONLY MODE:
# Path to the directory where your model training results were saved (e.g., 'runs/detect/pcb_component_model').
# This folder MUST contain a 'weights/best.pt' file.
LAST_RUN_PATH = '/content/gdrive/MyDrive/Colab Notebooks/project3_data/runs/detect/pcb_component_model'
# ==========================================

# ==========================================
# 0. SETUP & PATH CONFIGURATION
# ==========================================
def setup_environment():
    # Mount Google Drive
    # Will only prompt if not already mounted
    if not os.path.exists('/content/gdrive/MyDrive'):
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

# --- New Function to Load Weights ---
def load_last_trained_model(run_path):
    print(f"\n--- Loading Trained Model from {run_path} ---")
    weights_path = os.path.join(run_path, 'weights', 'best.pt')
    
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at: {weights_path}")
        print("Please check if LAST_RUN_PATH is correct and training ran successfully before.")
        return None
    
    # Load the model directly from the weights file
    model = YOLO(weights_path)
    print("Model loaded successfully.")
    return model

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
    # The current threshold (110) and morphological operations are optimized for the provided image.
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
    model = YOLO('yolo11n.pt') 

    # HYPERPARAMETERS (Based on PDF requirements)
    # epochs: Must be below 200
    # imgsz: Minimum 900
    # batch: Reduced to 8 to prevent RAM crashes on Colab T4 GPU
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
        plots=True,    # Automatically generates confusion matrix & curves
        workers=2      # LOWERS RAM USAGE: Limits data loading to 2 CPU cores
    )
    
    print("Training Complete.")
    # The last run path is created from project and name
    final_run_path = os.path.join(PROJECT_PATH, NAME)
    return model, final_run_path

# ==========================================
# 3. PART 3: EVALUATION
# ==========================================
def step3_evaluation(model, base_dir, save_dir):
    print("\n--- Starting Step 3: Evaluation ---")
    
    # Path to evaluation images
    eval_folder = os.path.join(base_dir, 'evaluation')
    
    if not os.path.exists(eval_folder):
        # This is where the error likely occurred previously.
        print(f"ERROR: Evaluation folder not found at {eval_folder}. Cannot run inference.")
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
        name='evaluation_results',
        exist_ok=True   # Allow existing files
    )

    print(f"Evaluation Complete. Results saved to a folder inside: {save_dir}/evaluation_results")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    base_directory = setup_environment()
    
    # Handle execution based on user-defined mode
    if MAIN_MODE == "FULL_RUN":
        print("Running FULL_RUN: Masking -> Training -> Evaluation")
        step1_object_masking(base_directory)
        trained_model, run_path = step2_yolo_training(base_directory)
        step3_evaluation(trained_model, base_directory, run_path)

    elif MAIN_MODE == "EVALUATION_ONLY":
        print(f"Running EVALUATION_ONLY. Loading model from: {LAST_RUN_PATH}")
        
        # 1. Load the model from the saved weights
        loaded_model = load_last_trained_model(LAST_RUN_PATH)

        if loaded_model:
            # 2. Run Evaluation
            # Note: We use LAST_RUN_PATH as the save_dir so evaluation results are stored alongside the training run.
            step3_evaluation(loaded_model, base_directory, LAST_RUN_PATH)
        
    else:
        print("Invalid MAIN_MODE specified. Set to 'FULL_RUN' or 'EVALUATION_ONLY'.")