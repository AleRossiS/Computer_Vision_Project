import os
import re
import glob
import json
import cv2
import numpy as np

'''
To rectify images from multiple cameras we used this script instead of the given one. This was done for simplicity since we already ahve the correct images to use in the roboflow dataset.
'''

def load_calibration(calib_path):
    """
    Loads the calibration parameters (camera matrix and distortion coefficients) from a JSON file.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
        
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    
    return mtx, dist

def get_undistort_maps(mtx, dist, image_size):
    """
    Computes the undistortion and rectification transformation maps efficiently for a specific camera.
    """
    width, height = image_size
    map_x, map_y = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx,
        distCoeffs=dist,
        R=None,
        newCameraMatrix=mtx,
        size=(width, height),
        m1type=cv2.CV_32FC1
    )
    return map_x, map_y


def main():
    # -------------------------------------------------------------
    # INPUT AND OUTPUT PATHS
    # -------------------------------------------------------------
    images_dir = "dataset_roboflow/hpe_01.v1i.coco"
    output_dir = "rectified_images/hpe_rect_01"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Search for all images starting with 'out' (.jpg or .jpeg)
    image_files = sorted(
        glob.glob(os.path.join(images_dir, "out*.jpg")) + 
        glob.glob(os.path.join(images_dir, "out*.jpeg"))
    )
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    # Dictionary to cache the remap coordinates per camera (improves performance)
    camera_maps = {} 
    print(f"Found {len(image_files)} images to process...")

    for img_path in image_files:
        basename = os.path.basename(img_path)
        
        # Extract camera index and frame number using regex
        # Matches strings like "out1_frame_0002_..." -> cam_index=1, frame_num=0002
        match = re.search(r'^out(\d+)_frame_(\d+)', basename)
        if not match:
            print(f"Unable to extract camera and frame from: {basename}")
            continue
            
        cam_index = match.group(1)
        frame_str = match.group(2)
        
        # Keep only the last two digits of the frame (e.g., '0002' -> '02')
        frame_short = f"{int(frame_str):02d}"
        
        # Generate the new, clean filename (e.g., out1_01.jpg, out1_02.jpg)
        new_filename = f"out{cam_index}_{frame_short}.jpg"

        # Path to the specific camera's calibration file
        calib_path = f"camera_data_with_Rvecs_2ndversion/Camera_config2/cam_{cam_index}/calib/camera_calib.json"

        if not os.path.exists(calib_path):
            print(f"Calibration file not found for cam {cam_index}: {calib_path}")
            continue

        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue

        height, width = img.shape[:2]

        # Compute and cache the maps only the first time for each camera
        if cam_index not in camera_maps:
            mtx, dist = load_calibration(calib_path)
            map_x, map_y = get_undistort_maps(mtx, dist, (width, height))
            camera_maps[cam_index] = (map_x, map_y)
            print(f"Rectification map generated for Camera {cam_index}")

        # Retrieve the cached maps
        map_x, map_y = camera_maps[cam_index]

        # Apply undistortion mapping
        rectified_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Save the rectified image with the new clean name
        output_path = os.path.join(output_dir, new_filename)
        cv2.imwrite(output_path, rectified_img)
        print(f"Saved: {basename} -> {new_filename}")

    print(f"\nProcess completed! Rectified images saved in: {output_dir}")

if __name__ == "__main__":
    main()