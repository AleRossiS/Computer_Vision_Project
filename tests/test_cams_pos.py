import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



'''This code is for visualizing the camera positions and orientations in a 3D space based on their calibration data. It reads the rotation and translation vectors from JSON files, computes the camera positions in world coordinates, and plots them along with their local axes.'''



# Base directory path and available camera IDs
BASE_DIR = "camera_data_with_Rvecs_2ndversion/Camera_config2"
#BASE_DIR = "camera_data_with_Rvecs/camera_data"
CAMERA_IDS = [1, 2, 3, 4, 5, 7]

def get_cam_geometry(rvec, tvec):
    """
    Calculate the camera position and its rotation matrix in world coordinates.
    """
    rvec = np.array(rvec, dtype=np.float64)
    tvec = np.array(tvec, dtype=np.float64)
    
    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)  
    
    # Inverse rotation matrix (R^T) represents the orientation of cam axes in world frame
    R_T = R.T
    
    # Calculate camera position in world coordinates: C = -R^T * T
    cam_pos_world = -np.dot(R_T, tvec) 
    
    # Flatten the position array and convert to meters
    pos_meters = cam_pos_world.flatten() / 1000.0
    
    return pos_meters, R_T
  
def main(): 
  # Dictionary to hold camera positions and orientations
  world_cams_data = {}
  
  # --- Loop through each camera ID and read its calibration data ---
  for cam_id in CAMERA_IDS:
    file_path = os.path.join(BASE_DIR, f"cam_{cam_id}", "calib", f"camera_calib.json")
    #file_path = os.path.join(BASE_DIR, f"cam_{cam_id}", "calib", f"camera_calib_real.json")
    
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found for Cam {cam_id}: {file_path}")
        continue
      
    try:
      with open(file_path, 'r') as f:
        data = json.load(f)
      
      rvec = data.get("rvecs")
      tvec = data.get("tvecs")
      
      if rvec is None or tvec is None:
          print(f"[WARNING] Missing rvec or tvec for Cam {cam_id} in {file_path}")
          continue
      
      # Calculate camera position and rotation matrix
      pos, R_T = get_cam_geometry(rvec, tvec)
      world_cams_data[cam_id] = (pos, R_T)
      
    except Exception as e:
      print(f"[ERROR] Failed to process Cam {cam_id}: {e}")
      
  # --- Plotting ---
  fig = plt.figure(figsize=(12, 9))
  ax = fig.add_subplot(111, projection='3d')

  # 1. Plot Global Origin Ground Truth Point
  ax.scatter(0, 0, 0, color='black', s=150, marker='*', label='Ground Truth', zorder=5)
  ax.text(0, 0, 0, "   ORIGIN ", color='black', fontsize=10, weight='bold')

  # 2. Plot Global Frame Axes (Lines) at (0,0,0)
  axis_length_global = 3.5  # Length of global lines in meters
  
  # Global X-axis (Red)
  ax.plot([0, axis_length_global], [0, 0], [0, 0], color='r', linewidth=3, label='X axis')
  # Global Y-axis (Green)
  ax.plot([0, 0], [0, axis_length_global], [0, 0], color='g', linewidth=3, label='Y axis')
  # Global Z-axis (Blue)
  ax.plot([0, 0], [0, 0], [0, axis_length_global], color='b', linewidth=3, label='Z axis')

  # Color map for camera identification dots
  cmap = plt.get_cmap('tab10')
  axis_length_cam = 2.5  # Normalized length of ALL camera axes in meters

  # 3. Loop through each camera data to plot positions and local frames
  for i, (cam_id, (pos, R_T)) in enumerate(world_cams_data.items()):
    color_cam = cmap(i % 10)
    
    # Plot the camera center point
    ax.scatter(pos[0], pos[1], pos[2], color=color_cam, s=150, marker='o')
    ax.text(pos[0], pos[1], pos[2], f"  Cam {cam_id}", color=color_cam, fontsize=10, weight='semibold')
    
    # Draw dashed line of sight pointing to origin
    ax.plot([pos[0], 0], [pos[1], 0], [pos[2], 0], color=color_cam, linestyle='--', alpha=0.3)
    
    # Extract local camera axes directions from R_T columns.
    # R_T is an orthogonal rotation matrix, so its columns are already unit vectors (normalized to 1).
    cam_x_dir = R_T[:, 0]
    cam_y_dir = R_T[:, 1]
    cam_z_dir = R_T[:, 2] # Optical axis
    
    # Plot Local X-axis (Red)
    ax.plot([pos[0], pos[0] + axis_length_cam * cam_x_dir[0]],
            [pos[1], pos[1] + axis_length_cam * cam_x_dir[1]],
            [pos[2], pos[2] + axis_length_cam * cam_x_dir[2]], color='r', linewidth=1.5)
            
    # Plot Local Y-axis (Green)
    ax.plot([pos[0], pos[0] + axis_length_cam * cam_y_dir[0]],
            [pos[1], pos[1] + axis_length_cam * cam_y_dir[1]],
            [pos[2], pos[2] + axis_length_cam * cam_y_dir[2]], color='g', linewidth=1.5)
            
    # Plot Local Z-axis / Optical Axis (Blue)
    ax.plot([pos[0], pos[0] + axis_length_cam * cam_z_dir[0]],
            [pos[1], pos[1] + axis_length_cam * cam_z_dir[1]],
            [pos[2], pos[2] + axis_length_cam * cam_z_dir[2]], color='b', linewidth=2.0)
      
  # Set axis labels
  ax.set_xlabel('X Global (m)', fontsize=11, labelpad=10)
  ax.set_ylabel('Y Global (m)', fontsize=11, labelpad=10)
  ax.set_zlabel('Z Global (m)', fontsize=11, labelpad=10)
  
  ax.legend(loc='upper left')
  
  # Safe check for older Matplotlib versions (< 3.3.0) Io ho quella vecchia mi sa
  if hasattr(ax, 'set_box_aspect'):
      ax.set_box_aspect((1, 1, 1))
  else:
      print("[INFO] Older Matplotlib version detected. Skipping 'set_box_aspect'.")
  
  # Dynamic axis scaling to keep everything centered inside the cube
  extracted_positions = [data[0] for data in world_cams_data.values()]
  all_points = np.array(extracted_positions + [[0, 0, 0]])
  max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
  mid_x, mid_y, mid_z = all_points.mean(axis=0)
  
  # Set identical ranges for all 3 axes to fake a 1:1:1 aspect ratio on older Matplotlib
  ax.set_xlim(mid_x - max_range, mid_x + max_range)
  ax.set_ylim(mid_y - max_range, mid_y + max_range)
  ax.set_zlim(mid_z - max_range, mid_z + max_range)

  plt.show()

if __name__ == "__main__":
  main()