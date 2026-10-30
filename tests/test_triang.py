import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re # used for extracting numbers from the json files


'''
This script reconstructs the 3D skeleton of a specific target player 
from a multi-camera setup using calibrated computer vision techniques.
'''



#========================
# Calibration Functions
#========================

def load_cam_params(json_path):
  ''' 
  Loads camera parameters from a JSON file and computes the intrinsic matrix K, distortion coefficients dist, rotation vector rvec, translation vector tvec, and camera matrix P.

  Args:
      json_path (str): Path to the JSON file containing camera parameters.

  Returns:
      P (np.ndarray): Camera matrix.
      K (np.ndarray): Intrinsic camera parameters matrix.
      dist (np.ndarray): Distortion coefficients.

  Note: The notation is from "Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision"
  '''
  
  # Load camera parameters from the JSON file 
  with open(json_path, 'r') as f:
    data = json.load(f)

    K = np.array(data['mtx']) # Intrinsic camera parameters matrix
    dist = np.array(data['dist']) # Distortion coefficients
    rvec = np.array(data['rvecs']) # Rotation vector (Rodrigues)
    tvec = np.array(data['tvecs']) # Translation vector

    R, _ = cv2.Rodrigues(rvec) # Convert rotation vector to rotation matrix
    Rt = np.hstack((R, tvec))  # Combine rotation matrix and translation vector into a single matrix
    P = np.dot(K, Rt) # Camera matrix = K[R|t]

    return P, K, dist
  

def load_all_cams_params(cams_id, base_dir):
  '''
  Loads camera parameters for all cameras specified in cams_id from the base directory.

  Args:
      cams_id (list): List of camera IDs.
      base_dir (str): Base directory containing camera parameter JSON files.

  Returns:
      dict: A dictionary where keys are camera IDs and values are tuples of (P, K, dist).
  '''
  
  P_dict, K_dict, dist_dict = {}, {}, {} # Dictionaries to hold camera parameters for each camera
  
  for cam_id in cams_id:
    json_path = os.path.join(base_dir, f"cam_{cam_id}", "calib", f"camera_calib.json")
    
    if not os.path.exists(json_path):
        print(f"[WARNING] File not found for Cam {cam_id}: {json_path}")
        continue
      
    try:
      P, K, dist = load_cam_params(json_path)
      P_dict[cam_id] = P
      K_dict[cam_id] = K
      dist_dict[cam_id] = dist
    except Exception as e:
      print(f"[ERROR] Failed to load parameters for Cam {cam_id}: {e}")
  
  return P_dict, K_dict, dist_dict


#==========================
# Data Extraction Functions
#==========================

def extract_keypoints_single_frame(coco_json_path, player_name, frame_num):
  '''
  Extracts keypoints for a specific player and frame from a COCO JSON file.

  Args:
      coco_json_path (str): Path to the COCO JSON file.
      player_name (str): The name of the player whose keypoints are to be extracted. Ex : 'Red_11'
      frame_num (int): The frame number for which keypoints are to be extracted.

  Returns:
      dict: A dictionary mapping camera IDs (e.g., '1', '2') to a flat list of keypoint data [x1, y1, ...] for the specified player. Example: {'1': [x1, y1, x2, y2, ...], '2': [x1, y1, x2, y2, ...]}
  '''

  # Load the COCO JSON data
  with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

  # Map the string name (e.g., 'Red_11') to its numerical category ID
  target_category_id = None
  for cat in coco_data.get('categories', []):
    if cat['name'] == player_name:
      target_category_id = cat['id']
      break

  # If the player name doesn't exist in the dataset categories, stop early
  if target_category_id is None:
    print(f"[WARNING] Player name '{player_name}' not found in COCO categories.")
    return {}


  target_images = {} # Dictionary to hold the target images for the specified player and frame
  
  # === Loop through all the images and extract the ones that match the specified player_id and frame_num ===
  for img in coco_data['images']:

    # 1. Extract the camera ID and frame number from the image file name using regex
    file_name = img.get('extra', {}).get('name', img['file_name']) # Get img name, ex: out2_frame_0001.png
    match = re.search(r'out(\d+)_frame_(\d+)', file_name) # Search for the pattern "outX_frame_YYYY" in file_name

    if match:
      cam_id = match.group(1)         # Extract the camera ID from the first group of the match variable
      curr_frame_num = int(match.group(2)) # Extract the frame number from the second group of the match variable

    # 2. Check if the current image matches the specified frame_num
      if curr_frame_num == frame_num:
        target_images[img['id']] = cam_id # Store the img ID and corresponding cam ID in the target_images
                                          # Ex: 1042: "1" => img ID 1042 corresponds to camera 1


  # === Extract keypoints for the specified player_id and frame from the JSON annotations field ===
  
  # Create a dictionary to hold the keypoints for each camera
  keypoints_cams = {}
  
  # Loop through the annotations in the COCO JSON data
  for ann in coco_data['annotations']:

    # Check if the annotation corresponds to the specified player_id and if it is in the target_images dictionary
    if ann['image_id'] in target_images and ann['category_id'] == target_category_id:
      cam_id = target_images[ann['image_id']] # Get the camera ID for the current annotation
      keypoints_cams[cam_id] = ann['keypoints'] # Store the keypoints for the current camera

  return keypoints_cams


#=====================================
# Geometry and Triangulation Functions
#=====================================

def rectify_skeleton(keypoints_list, K, dist):
  '''
  Rectifies a list of keypoints using the camera's intrinsic matrix K and distortion coefficients dist.

  Args:
      keypoints_list (list): Flat list of 54 values [x1, y1, v1, x2, y2, v2...].
      K (np.ndarray): The intrinsic camera parameters matrix.
      dist (np.ndarray): The distortion coefficients.

  Returns:
      numpy.ndarray: Array of shape (18, 2) containing the undistorted (X, Y) coordinates.
  '''

  # Convert the flat list of keypoints into a numpy array and reshape it to (18, 3) for 18 keypoints with (x, y, v)
  keypoints_array = np.array(keypoints_list, dtype=np.float32).reshape(-1, 3)

  # Extract the (x, y) coordinates of the keypoints
  points_2d = keypoints_array[:, :2] # Get the first two columns (the third column is the visibility flag)
  visibility = keypoints_array[:, 2:3] # Get the visibility flag for each keypoint

  points_2d = points_2d.reshape(-1, 1, 2) # Reshape to (N, 1, 2) for cv2.undistortPoints

  # Undistort the points 
  rectified_points = cv2.undistortPoints(points_2d, K, dist, P=K).reshape(-1, 2) 

  # Return the rectified points with visibility flag as an array of shape (18, 3) -> (X, Y, visibility)
  return np.hstack((rectified_points, visibility))


def triangulate_points_svd(P_matrices, points):
  '''
  Solves the linear system AX = 0 to find the 3D coordinate (X, Y, Z)
  from N 2D views using SVD.    

  See "Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision" for more details.

  Args:
      P_matrices (list): List of camera projection matrices (P) for each view.
      points (list): List of corresponding 2D points in each view.

  Returns:
      numpy.ndarray: 3D point in homogeneous coordinates (X, Y, Z, W).
  '''

  A = []

  # Build the linear system AX = 0 for each camera view and corresponding 2D point
  for P, point in zip(P_matrices, points):
    x, y = point
    A.append(x * P[2, :] - P[0, :])  # First row of the linear system
    A.append(y * P[2, :] - P[1, :])  # Second row of the linear system
  
  A = np.array(A) # Convert the list of equations into a numpy array

  # Solve the linear system using SVD
  _, _, Vt = np.linalg.svd(A) # Perform Singular Value Decomposition
  X = Vt[-1] # The solution is the last row of Vt 

  # Return normalized homogeneous coordinates to get (X, Y, Z) 
  return (X / X[3])[:3] # Divide by the last coordinate to convert from homogeneous to Cartesian


def reconstruct_3d_skeleton(keypoints_cams, P_dict, K_dict, dist_dict):
  '''
  Reconstructs the 3D skeleton from 2D keypoints across multiple camera views.

  Args:
      keypoints_cams (dict): Dictionary mapping camera IDs to their corresponding 2D keypoints.
      P_dict (dict): Dictionary mapping camera IDs to their projection matrices.
      K_dict (dict): Dictionary mapping camera IDs to their intrinsic matrices.
      dist_dict (dict): Dictionary mapping camera IDs to their distortion coefficients.

  Returns:
    list: A list of 18 tuples (X, Y, Z) representing the 3D coordinates of the skeleton joints. 
    Returns None for joints that could not be triangulated (seen in less than 2 cameras).
      
  '''

  N_KEYPOINTS = 18
  skeleton_3d = []

  # 1. Rectify keypoints for each camera
  rectified_keypoints = {}
  for cam_id, keypoints in keypoints_cams.items():
    rectified_keypoints[cam_id] = rectify_skeleton(keypoints, K_dict[cam_id], dist_dict[cam_id])

  # 2. Triangulate each keypoint across all cameras
  for i in range(N_KEYPOINTS):
    points_2d = []
    P_matrices = []

    # Collect 2D points and corresponding projection matrices for the current keypoint across all cameras
    for cam_id in rectified_keypoints:

      # Extract X, Y and visibility for the current keypoint
      x = rectified_keypoints[cam_id][i, 0]
      y = rectified_keypoints[cam_id][i, 1]
      v = rectified_keypoints[cam_id][i, 2]

      if v > 0:  # Only consider keypoints that are visible
        points_2d.append((x, y)) # Append the 2D point to the list
        P_matrices.append(P_dict[cam_id]) # Append the corresponding projection matrix

    # Triangulate the 3D point if we have at least two views
    if len(points_2d) >= 2:
      coord_3d = triangulate_points_svd(P_matrices, points_2d)
      # coord_3d = triangulate_points_bundle_adjustment(P_matrices, points_2d, x0=coord_3d)
      skeleton_3d.append(tuple(coord_3d))
    else:
      skeleton_3d.append(None) # Fallback for keypoints that cannot be triangulated (seen in less than 2 cameras)

  return skeleton_3d


#===================
# Plotting Functions
#===================

def plot_3d_skeleton(skeleton_3d):
  '''
  Plots the 3D skeleton using Matplotlib.

  Args:
      skeleton_3d (list): A list of 18 tuples (X, Y, Z) representing the 3D coordinates of the skeleton joints.
  '''

  if not skeleton_3d or all(pt is None for pt in skeleton_3d):
    print("[WARNING] No valid 3D points to plot.")
    return
  
  # initialize the 3D plot
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')

  # Define the bones connections based on the 18-points, order:
  # 0:Head, 1:Neck, 2:RShoulder, 3:LShoulder, 4:RElbow, 5:LElbow, 6:RHand, 7:LHand, 
  # 8:Spine, 9:Hips, 10:RHip, 11:LHip, 12:RKnee, 13:LKnee, 14:RAnkle, 15:LAnkle, 16:RFoot, 17:LFoot
  bones = [
    (0, 1),   # Head - Neck
    (1, 2),   # Neck - Right Shoulder
    (1, 3),   # Neck - Left Shoulder
    (2, 4),   # Right Shoulder - Right Elbow
    (4, 6),   # Right Elbow - Right Hand
    (3, 5),   # Left Shoulder - Left Elbow
    (5, 7),   # Left Elbow - Left Hand
    (1, 8),   # Neck - Spine
    (8, 9),   # Spine - Hips
    (9, 10),  # Hips - Right Hip
    (9, 11),  # Hips - Left Hip
    (10, 12), # Right Hip - Right Knee
    (12, 14), # Right Knee - Right Ankle
    (14, 16), # Right Ankle - Right Foot
    (11, 13), # Left Hip - Left Knee
    (13, 15), # Left Knee - Left Ankle
    (15, 17)  # Left Ankle - Left Foot
  ]

  # Extract X, Y, Z for valid points to plot the scatter (the joints)
  valid_x, valid_y, valid_z = [], [], []
  for pt in skeleton_3d:
    if pt is not None:
      valid_x.append(pt[0])
      valid_y.append(pt[1])
      valid_z.append(pt[2])

  # Draw the joints (blue dots)
  ax.scatter(valid_x, valid_y, valid_z, c='b', marker='o', s=50, label='3D Joints')

  # Draw the bones (red lines) connecting valid joints
  for bone in bones:
    joint1_idx, joint2_idx = bone
    pt1 = skeleton_3d[joint1_idx]
    pt2 = skeleton_3d[joint2_idx]

    # Draw the line only if both points of the bone exist
    if pt1 is not None and pt2 is not None:
      # Create coordinate lists for the line: [X1, X2], [Y1, Y2], [Z1, Z2]
      xs = [pt1[0], pt2[0]]
      ys = [pt1[1], pt2[1]]
      zs = [pt1[2], pt2[2]]
      ax.plot(xs, ys, zs, c='r', linewidth=2)

  # Plot aesthetic configurations
  ax.set_title("3D Skeleton Reconstruction - Multi-Camera", fontsize=14, fontweight='bold')
  ax.set_xlabel('X Axis (mm)')
  ax.set_ylabel('Y Axis (mm)')
  ax.set_zlabel('Z Axis (mm)')
  ax.set_aspect('equal')

  # Show the plot on screen
  plt.legend()
  plt.show()


#===================
# Reprojecting from 3D to 2D
#===================


def reproject_3d_to_2d(skeleton_3d, P_dict):
    '''
    Projects 3D points into 2D space for each camera using P_dict.
    Returns a dictionary: {cam_id: np.array of shape (18, 2)}
    '''
    reprojected_cams = {}
    
    for cam_id, P in P_dict.items():
        reprojected_pts = []
        for pt_3d in skeleton_3d:
            if pt_3d is None:
                reprojected_pts.append([np.nan, np.nan])
                continue
            
            # 3D homogeneous coordinates [X, Y, Z, 1]
            X_hom = np.array([pt_3d[0], pt_3d[1], pt_3d[2], 1.0])
            
            # 2D homogeneous projection: p_2d_hom = P * X_hom
            p_2d_hom = np.dot(P, X_hom)
            
            # Convert from homogeneous to 2D Cartesian coordinates (divide by Z)
            x_2d = p_2d_hom[0] / p_2d_hom[2]
            y_2d = p_2d_hom[1] / p_2d_hom[2]
            
            reprojected_pts.append([x_2d, y_2d])
            
        reprojected_cams[cam_id] = np.array(reprojected_pts)
        
    return reprojected_cams


def plot_3d_players_on_court(
    target_players, 
    frame_num, 
    coco_json_path, 
    P_dict,
    K_dict,
    dist_dict,
    reconstruct_func, 
    skeleton_connections=None,
    court_dimensions=(28.0, 15.0)
):
    '''
    Reconstruct and visualize in the 3D basketball court the selected players
    '''

    # Definition of the skeleton connections if not defined
    if skeleton_connections is None:
        # 0: Head, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist,
        # 8: RHip, 9: RKnee, 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye, 15: LEye, 16: REar, 17: LEar
        # 0: Head, 1: Neck, 2: RShoulder, 3: LShoulder, 4: RElbow, 5: LElbow, 6: RHand, 7: LHand, 
        # 8: Spine, 9: Hips, 10: RHip, 11: LHip, 12: RKnee, 13: LKnee, 14: RAnkle, 15: LAnkle, 16: RFoot, 17: LFoot
        skeleton_connections = [
            (0, 1),   # Head - Neck
            (1, 2),   # Neck - Right Shoulder
            (1, 3),   # Neck - Left Shoulder
            (2, 4),   # Right Shoulder - Right Elbow
            (4, 6),   # Right Elbow - Right Hand
            (3, 5),   # Left Shoulder - Left Elbow
            (5, 7),   # Left Elbow - Left Hand
            (1, 8),   # Neck - Spine
            (8, 9),   # Spine - Hips
            (9, 10),  # Hips - Right Hip
            (9, 11),  # Hips - Left Hip
            (10, 12), # Right Hip - Right Knee
            (12, 14), # Right Knee - Right Ankle
            (14, 16), # Right Ankle - Right Foot
            (11, 13), # Left Hip - Left Knee
            (13, 15), # Left Knee - Left Ankle
            (15, 17)  # Left Ankle - Left Foot
        ]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the lines of the basketball court
    length, width = court_dimensions
    court_lines = [
        # External perimeter
        [[-length/2, length/2, length/2, -length/2, -length/2], 
         [-width/2, -width/2, width/2, width/2, -width/2], 
         [0, 0, 0, 0, 0]],
        # Mid court line
        [[0, 0], [-width/2, width/2], [0, 0]]
    ]

    for line in court_lines:
        ax.plot(line[0], line[1], line[2], color='red', linestyle='--', alpha=0.6)

    # Function that maps the colors of players
    def get_player_color(name):
        if name.startswith('Red'):
            return 'red'
        elif name.startswith('White'):
            return 'blue'
        else:
            return 'green'

    # Reconstruct players positions
    players_found = 0

    for player in target_players:
        # 1) Extract 2D keypoints for the player in that specific frame
        keypoints_2d_cams = extract_keypoints_single_frame(coco_json_path, player, frame_num)

        if not keypoints_2d_cams:
            print(f"[INFO] No 2D data for {player} at frame {frame_num}.")
            continue

        # 2) Calling the selected triangulation function passing the 2D keypoint dictionary and calibration parameters
        keypoints_3d = reconstruct_func(keypoints_2d_cams, P_dict, K_dict, dist_dict)

        if keypoints_3d is None or all(pt is None for pt in keypoints_3d):
            print(f"[INFO] Failed reconstruction for {player} at frame {frame_num}.")
            continue

        # Convert None points to np.nan for correct numpy management
        kpts_array = np.array([pt if pt is not None else [np.nan, np.nan, np.nan] for pt in keypoints_3d])

        # 3) Convert mm to meters in order to visualize
        kpts_array = kpts_array / 1000.0
        color = get_player_color(player)
        players_found += 1

        # 4) Plot the joints in 3D
        valid_mask = ~np.isnan(kpts_array[:, 0])
        ax.scatter(
            kpts_array[valid_mask, 0], 
            kpts_array[valid_mask, 1], 
            kpts_array[valid_mask, 2], 
            color=color, 
            s=15, 
            label=f'{player}'
        )

        # 5) Plot the connections between the joints
        for p1_idx, p2_idx in skeleton_connections:
            if p1_idx < len(kpts_array) and p2_idx < len(kpts_array):
                pt1 = kpts_array[p1_idx]
                pt2 = kpts_array[p2_idx]

                if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                    ax.plot(
                        [pt1[0], pt2[0]], 
                        [pt1[1], pt2[1]], 
                        [pt1[2], pt2[2]], 
                        color=color, 
                        linewidth=2
                    )

        # 6) Label the name of each player above its head (its first valid joint)
        head_pos = kpts_array[0] if not np.isnan(kpts_array[0]).any() else kpts_array[valid_mask][0]
        ax.text(head_pos[0], head_pos[1], head_pos[2] + 0.15, player, color=color, fontweight='bold', fontsize=9)

    # PLOT OPTIONS CONFIGURATION
    ax.set_title(f"3D Skeletons Reconstruction - Frame {frame_num}", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


    # Let's visualize only where the players are in order to visualize them bigger
    ax.set_xlim([0, 14.5])     # Half court that is empty
    ax.set_ylim([-8, 8])       # Shrinks Y around players
    ax.set_zlim([0, 2.2])    

    # Box Aspect option
    ax.set_box_aspect((15, 12, 1.5))

    # View inclination
    ax.view_init(elev=22, azim=-55)

    # ZOOM: in new versions of MatPlotLib
    try:
        ax.set_zoom(1.4)  # +40% Zoom
    except AttributeError:
        ax.dist = 7      # For older versions

    plt.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=1.0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.show()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_3d_players_on_court(
    target_players, 
    start_frame, 
    end_frame, 
    coco_json_path, 
    P_dict, 
    K_dict, 
    dist_dict, 
    reconstruct_func, 
    skeleton_connections=None,
    court_dimensions=(28.0, 15.0),
    interval=100 # Milliseconds between frames (100ms = 10 FPS)
):
    if skeleton_connections is None:
        skeleton_connections = [
            (0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),
            (1, 8), (8, 9), (9, 10), (9, 11), (10, 12), (12, 14), 
            (14, 16), (11, 13), (13, 15), (15, 17)
        ]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw the static basketball court
    length, width = court_dimensions
    court_lines = [
        [[-length/2, length/2, length/2, -length/2, -length/2], 
         [-width/2, -width/2, width/2, width/2, -width/2], 
         [0, 0, 0, 0, 0]],
        [[0, 0], [-width/2, width/2], [0, 0]]
    ]
    for line in court_lines:
        ax.plot(line[0], line[1], line[2], color='gray', linestyle='--', alpha=0.5)

    def get_player_color(name):
        return 'red' if name.startswith('Red') else ('blue' if name.startswith('White') else 'green')

    # 2. Initialize graphic objects (Scatter, Lines, and Text for each player)
    player_graphics = {}
    for player in target_players:
        color = get_player_color(player)
        
        # 3D Points (Scatter)
        scat = ax.scatter([], [], [], color=color, s=25, label=player)
        
        # Bones (Lines)
        lines = [ax.plot([], [], [], color=color, linewidth=2)[0] for _ in skeleton_connections]
        
        # Name Label
        txt = ax.text(0, 0, 0, player, color=color, fontweight='bold', fontsize=8)
        
        player_graphics[player] = {
            'scat': scat,
            'lines': lines,
            'txt': txt
        }

    # 3. Initial Camera and Limits Settings (Zoom on the action)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    # Optimal limits and zoom (cropped to the action area)
    ax.set_xlim([-1, 14])
    ax.set_ylim([-8, 8])
    ax.set_zlim([0, 2.2])
    ax.set_box_aspect((15, 12, 1.5))
    ax.view_init(elev=22, azim=-55)

    try:
        ax.set_zoom(1.3)
    except AttributeError:
        ax.dist = 7

    plt.subplots_adjust(left=-0.1, right=1.1, bottom=-0.1, top=0.95)

    # Remove legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # 4. Update function called at each Frame
    def update(frame):
        ax.set_title(f"3D Skeleton Reconstruction - Frame {frame}", fontsize=14)

        for player in target_players:
            gfx = player_graphics[player]
            
            # Extraction and 3D Reconstruction for the current frame
            keypoints_2d = extract_keypoints_single_frame(coco_json_path, player, frame)
            
            if not keypoints_2d:
                # Hide elements if the player is not present in this frame
                gfx['scat']._offsets3d = ([], [], [])
                for line in gfx['lines']:
                    line.set_data([], [])
                    line.set_3d_properties([])
                gfx['txt'].set_text('')
                continue

            kpts_3d = reconstruct_func(keypoints_2d, P_dict, K_dict, dist_dict)
            if kpts_3d is None or all(pt is None for pt in kpts_3d):
                continue

            # Convert to array and scale from mm to Meters
            kpts_array = np.array([pt if pt is not None else [np.nan, np.nan, np.nan] for pt in kpts_3d]) / 1000.0

            valid_mask = ~np.isnan(kpts_array[:, 0])
            valid_pts = kpts_array[valid_mask]

            if len(valid_pts) == 0:
                continue

            # Update Points (Joints)
            gfx['scat']._offsets3d = (valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2])

            # Update Bones (Lines)
            for idx, (p1_idx, p2_idx) in enumerate(skeleton_connections):
                if p1_idx < len(kpts_array) and p2_idx < len(kpts_array):
                    pt1, pt2 = kpts_array[p1_idx], kpts_array[p2_idx]
                    if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                        gfx['lines'][idx].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])
                        gfx['lines'][idx].set_3d_properties([pt1[2], pt2[2]])
                    else:
                        gfx['lines'][idx].set_data([], [])
                        gfx['lines'][idx].set_3d_properties([])

            # Update Name Label above the head
            head_pos = valid_pts[0]
            gfx['txt'].set_position_3d((head_pos[0], head_pos[1], head_pos[2] + 0.15))
            gfx['txt'].set_text(player)

    # 5. Start Animation
    frames_range = range(start_frame, end_frame + 1)
    # It is CRUCIAL to save the instance in an 'anim' variable to prevent the Garbage Collector from stopping it
    anim = FuncAnimation(fig, update, frames=frames_range, interval=interval, repeat=True)

    plt.show()
    return anim


#===================
# Evaluate Reprojection Error
#===================

def evaluate_reprojection_error(keypoints_cams, reprojected_cams, K_dict, dist_dict):
    '''
    Computes the Mean Per Joint Position Error (MPJPE) in pixels
    and the Mean Squared Error (MSE) between the rectified GT points and the reprojected ones.
    '''
    total_errors = []
    
    # Prepare the rectified GT points
    rectified_gt = {}
    for cam_id, kpts in keypoints_cams.items():
        rectified_gt[cam_id] = rectify_skeleton(kpts, K_dict[cam_id], dist_dict[cam_id])

    for cam_id in keypoints_cams:
        gt_pts = rectified_gt[cam_id]               # Array (18, 3) -> [x, y, vis]
        proj_pts = reprojected_cams[cam_id]         # Array (18, 2)
        
        for i in range(18):
            v = gt_pts[i, 2] # Visibility
            if v > 0 and not np.isnan(proj_pts[i, 0]):
                gt_xy = gt_pts[i, :2]
                proj_xy = proj_pts[i]
                
                # Euclidean distance in pixels for this joint
                err = np.linalg.norm(gt_xy - proj_xy)
                total_errors.append(err)
                
    if len(total_errors) == 0:
        return 0.0, 0.0
    
    # MPJPE = mean of the Euclidean distances
    mpjpe_2d = np.mean(total_errors)
    # MSE = mean of the squared errors
    mse_2d = np.mean(np.square(total_errors))
    
    return mpjpe_2d, mse_2d





if __name__ == "__main__":
    
  # ---------------------------------------------------------
  # 1. PARAMETER AND PATH CONFIGURATION
  # ---------------------------------------------------------
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  BASE_CALIB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "camera_data_with_Rvecs_2ndversion", "Camera_config2"))
  COCO_JSON_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "dataset_roboflow", "hpe_01.v1i.coco", "_annotations.coco.json"))
  
  # Define available cameras in the setup
  CAMS_ID = ["1", "2", "3", "4", "5", "7"]
  
  # Set the target to analyze
  # Red_11, Red_13, Red_23, Red_7, Red_9
  # White_10, White_11, White_13, White_22, White_27
  TARGET_PLAYER = "Red_9"   # Name of the player whose keypoints are to be extracted
  TARGET_FRAME = 1          # The video frame you want to reconstruct in 3D

  print("\n" + "="*40)
  print("--- STARTING 3D PIPELINE ---")
  print(f"Target: Player {TARGET_PLAYER} | Frame: {TARGET_FRAME}")
  print("="*40)

  # ---------------------------------------------------------
  # 2. LOADING CALIBRATIONS AND MATRICES
  # ---------------------------------------------------------
  print("\n[INFO] [1/4] Loading camera parameters...")
  P_dict, K_dict, dist_dict = load_all_cams_params(CAMS_ID, BASE_CALIB_DIR)
  
  if not P_dict:
      print("[CRITICAL ERROR] No calibrations loaded. Aborting.")
      exit()

  # ---------------------------------------------------------
  # 3. 2D DATA EXTRACTION FROM JSON
  # ---------------------------------------------------------
  print("\n[INFO] [2/4] Extracting 2D keypoints from COCO dataset...")
  keypoints_2d_cams = extract_keypoints_single_frame(COCO_JSON_PATH, TARGET_PLAYER, TARGET_FRAME)

  if not keypoints_2d_cams:
      print(f"[WARNING] Player {TARGET_PLAYER} was not found in any camera at frame {TARGET_FRAME}.")
      exit()
      
  print(f"       -> Player detected in {len(keypoints_2d_cams)} cameras.")

  # ---------------------------------------------------------
  # 4. 3D TRIANGULATION AND VISUALIZATION
  # ---------------------------------------------------------
  print("\n[INFO] [3/4] 3D Reconstruction (Undistort + SVD Triangulation)...")
  final_3d_skeleton = reconstruct_3d_skeleton(keypoints_2d_cams, P_dict, K_dict, dist_dict)

  # Count how many points were successfully triangulated
  valid_points = sum(1 for pt in final_3d_skeleton if pt is not None)
  print(f"       -> Triangulation completed: {valid_points}/18 joints reconstructed.")

  # Launch graphical visualization
  print("\n[INFO] Generating 3D plot...")
  plot_3d_skeleton(final_3d_skeleton)


  # ---------------------------------------------------------
  # 5. REPROJECTION AND PERFORMANCE EVALUATION
  # ---------------------------------------------------------
  print("\n[INFO] [4/4] Reprojecting and Evaluating Performance...")
  reprojected_2d = reproject_3d_to_2d(final_3d_skeleton, P_dict)
  
  mpjpe_err, mse_err = evaluate_reprojection_error(
      keypoints_2d_cams, 
      reprojected_2d, 
      K_dict, 
      dist_dict
  )
  
  print(f"\n--- EVALUATION RESULTS ---")
  print(f"Mean Per Joint Position Error (MPJPE): {mpjpe_err:.3f} px")
  print(f"Mean Squared Error (MSE):              {mse_err:.3f} px^2")
  print("\n" + "="*40)
  print("--- PIPELINE FINISHED ---")
  print("="*40 + "\n")


  # Target all the players
  TARGET_PLAYERS = [
      'Red_11', 'Red_13', 'Red_23', 'Red_7', 'Red_9',
      'White_10', 'White_11', 'White_13', 'White_22', 'White_27'
  ]

  # Execute the visualization for all the players on court
  plot_3d_players_on_court(
      target_players=TARGET_PLAYERS,
      frame_num=TARGET_FRAME,
      coco_json_path=COCO_JSON_PATH,
      P_dict=P_dict,
      K_dict=K_dict,
      dist_dict=dist_dict,
      reconstruct_func=reconstruct_3d_skeleton
  )

  # Execute the animation for all the frames
  anim = animate_3d_players_on_court(
      target_players=TARGET_PLAYERS,
      start_frame=1,                  # Initial frame
      end_frame=5,                   # Final frame
      coco_json_path=COCO_JSON_PATH,
      P_dict=P_dict,
      K_dict=K_dict,
      dist_dict=dist_dict,
      reconstruct_func=reconstruct_3d_skeleton,
      interval=200                     # milliseconds ~ 5 FPS
  )