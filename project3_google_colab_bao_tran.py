import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys

# --- AUTO-INSTALL ULTRALYTICS ---
# Ensures YOLOv11 libraries are present in the Colab runtime
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
    # Mount Google Drive to access your dataset
    drive.mount('/content/gdrive')

    BASE_DIR = '/content/gdrive/MyDrive/Colab Notebooks/project3_data'

    if not os.path.exists(BASE_DIR):
        print(f"WARNING: Directory {BASE_DIR} not found. Please check your Drive paths.")
    else:
        print(f"Working directory set to: {BASE_DIR}")

    return BASE_DIR

# ==========================================
# 1. PART 1: OBJECT MASKING
# ==========================================
def step1_object_masking(base_dir):
    print("\n--- Starting Step 1: Object Masking ---")

    img_path = os.path.join(base_dir, 'motherboard_image.JPEG')

    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 1. Read the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Pre-processing
    # Convert to grayscale and apply Gaussian Blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Edge Detection (Canny)
    edges = cv2.Canny(blur, 50, 150)

    # 4. Morphological Operations
    # Dilate the edges to connect gaps and form a solid boundary
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Close operation to fill inside the board area if needed
    closed_mask = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # 5. Find Contours
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filter: The motherboard is the largest object in the scene
        largest_contour = max(contours, key=cv2.contourArea)

        # Create the final clean mask
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # 6. Extract the PCB using bitwise_and
        result = cv2.bitwise_and(img, img, mask=final_mask)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # 7. Visualization
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Canny Edges (Dilated)")
        plt.imshow(closed_mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Extracted PCB")
        plt.imshow(result_rgb)
        plt.axis('off')

        plt.show()

        # Save the result
        output_path = os.path.join(base_dir, 'masked_motherboard_result.jpg')
        cv2.imwrite(output_path, result)
        print(f"Masked image saved to: {output_path}")

    else:
        print("No contours found. Check Canny parameters.")

# ==========================================
# 2. PART 2: YOLOv11 TRAINING (OPTIMIZED)
# ==========================================
def step2_yolo_training(base_dir):
    print("\n--- Starting Step 2: YOLOv11 Training ---")

    yaml_path = os.path.join(base_dir, 'data.yaml')

    # Using YOLOv11 Nano (yolo11n.pt) as recommended
    model = YOLO('yolo11n.pt')

    # HYPERPARAMETERS
    EPOCHS = 150
    IMG_SIZE = 900
    BATCH_SIZE = 8  # Kept at 8 to be safe for T4 GPU memory
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
        exist_ok=True,
        plots=True,       # Saves confusion matrix and curves for your report
        workers=2,        # Low worker count to save RAM
        patience=25,      # Early stopping
        augment=True      # Data augmentation to improve robustness
    )

    print("Training Complete.")
    return model, os.path.join(PROJECT_PATH, NAME)

# ==========================================
# 3. PART 3: EVALUATION
# ==========================================
def step3_evaluation(model, base_dir, save_dir):
    print("\n--- Starting Step 3: Evaluation ---")

    eval_folder = os.path.join(base_dir, 'evaluation')

    if not os.path.exists(eval_folder):
        print(f"Evaluation folder not found at {eval_folder}")
        return

    # Run Inference
    # Using conf=0.20 (slightly lower) to ensure small components are detected
    results = model.predict(
        source=eval_folder,
        imgsz=900,
        conf=0.20,
        iou=0.45,
        save=True,
        project=save_dir,
        name='evaluation_results'
    )

    print(f"Evaluation results saved to: {save_dir}/evaluation_results")

    # Optional: Display results inline
    for result in results:
        # Plot results (BGR to RGB for matplotlib)
        im_array = result.plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup
    base_directory = setup_environment()

    # 2. Run Masking (Improved with Canny)
    step1_object_masking(base_directory)

    # 3. Run Training (Optimized for T4)
    trained_model, run_path = step2_yolo_training(base_directory)

    # 4. Run Evaluation
    step3_evaluation(trained_model, base_directory, run_path)