import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Sanity Check Script: 3D to 2D Reprojection
------------------------------------------
This script loads known 3D court points and their 2D correspondences.
It then project the 3D points back onto the 2D image plane and visually compares them 
against the 2D correspondences.

Uses the data in the cameras JSON files img_points.json. 
'''

# ==========================================
# 1. CONFIGURATION PATHS
# ==========================================
# Base directories (Adjust to change the camera or dataset)
#CALIB_DIR = 'camera_data_with_Rvecs_2ndversion/Camera_config2/cam_3/calib'
CALIB_DIR = 'camera_data_with_Rvecs/camera_data/cam_3/calib'
  
# Paths to the JSON files
camera_calib_json = os.path.join(CALIB_DIR, 'camera_calib.json')
img_points_json = os.path.join(CALIB_DIR, 'img_points.json')

# Path to the image corresponding to cam_3
# (Replace with the correct path to your cam_3 image)
image_path = 'dataset_roboflow/hpe_01.v1i.coco/out3_frame_0001_png.rf.0bab75e5e390766f75af8a70f768a56d.jpg'
#image_path = 'rectified_images/hpe_rect_01/out3_01.jpg'


# ==========================================
# 2. LOAD CALIBRATION AND POINT DATA
# ==========================================
print("[INFO] Loading data...")

# Load Camera Calibration Parameters (Intrinsics & Extrinsics)
with open(camera_calib_json, 'r') as f:
    calib_data = json.load(f)
    K = np.array(calib_data['mtx'], dtype=np.float32)      # Intrinsic matrix
    dist = np.array(calib_data['dist'], dtype=np.float32)  # Distortion coefficients
    rvec = np.array(calib_data['rvecs'], dtype=np.float32) # Rotation vector
    tvec = np.array(calib_data['tvecs'], dtype=np.float32) # Translation vector

# Load Points from img_points.json
with open(img_points_json, 'r') as f:
    corners_data = json.load(f)
    # Extract the lists from the JSON 
    points_3d_world = corners_data['real_corners']
    points_2d = corners_data['img_corners']

# Convert lists to NumPy arrays of type float32 (Strict OpenCV requirement)
points_3d_array = np.array(points_3d_world, dtype=np.float32)
points_2d_array = np.array(points_2d, dtype=np.float32)

points_3d_array = points_3d_array * 1000.0  # Convert meters to millimeters for consistency

print(f"[INFO] Loaded {len(points_3d_array)} matched points.")


# ==========================================
# 3. REPROJECT 3D POINTS TO 2D
# ==========================================
print("[INFO] Computing reprojections...")

# cv2.projectPoints applies K, rvec, tvec AND the distortion coefficients ('dist')
projected_3d_points, _ = cv2.projectPoints(points_3d_array, rvec, tvec, K, dist)

# Reshape from OpenCV's tricky (N, 1, 2) format to a clean (N, 2) array
projected_3d_points = projected_3d_points.reshape(-1, 2)


# ==========================================
# 4. CALCULATE ERROR
# ==========================================
# Calculate Euclidean distance between projected points and manual ground truth points
errors = []
for i in range(len(points_2d_array)):
    gt_pt = points_2d_array[i]
    proj_pt = projected_3d_points[i]
    # np.linalg.norm computes the straight-line distance in pixels
    dist_err = np.linalg.norm(gt_pt - proj_pt)
    errors.append(dist_err)

mean_error = np.mean(errors)
print(f"[RESULT] Average Reprojection Error: {mean_error:.2f} pixels")


# ==========================================
# 5. VISUALIZATION
# ==========================================
print("[INFO] Rendering plot...")

# Load and convert image from BGR to RGB for Matplotlib
img = cv2.imread(image_path)
if img is None:
    print(f"[ERROR] Image not found at: {image_path}")
    exit()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Setup figure
fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(img)

# Plot elements loop
for i in range(len(points_2d_array)):
    # Get coordinates
    gt_x, gt_y = points_2d_array[i]
    proj_x, proj_y = projected_3d_points[i]
    
    # Plot Ground Truth (Manual Annotations) as Red X's
    ax.plot(gt_x, gt_y, 'rx', markersize=12, markeredgewidth=2.5, 
            label='Original 2D' if i == 0 else "")
    
    # Plot Reprojected Points as Green Dots
    ax.plot(proj_x, proj_y, 'go', markersize=8, markeredgecolor='white', 
            label='Reprojected 3D' if i == 0 else "")
    
    # Draw a dashed yellow line connecting them to visualize the error offset
    ax.plot([gt_x, proj_x], [gt_y, proj_y], 'y--', alpha=0.8, linewidth=1.5)
    
    # Add a small text label (P0, P1, P2...) to identify specific problematic points
    ax.text(proj_x + 15, proj_y - 15, f"P{i}", color='cyan', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

# Configure Plot UI
ax.set_title(f"Sanity Check: Cam 3 | Avg Error: {mean_error:.2f} px", fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=12)
ax.axis('off') # Hide axis ticks
plt.tight_layout()

# Show the interactive window
plt.show()