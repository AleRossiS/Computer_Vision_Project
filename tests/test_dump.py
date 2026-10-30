import cv2
import os
import json



'''This script is just to see what the dump file contains and to visualize the points on a video.
Apparently it contains the point representing the calibaration grid pattern in the video frames.'''



def extract_points_from_json(json_file):
  '''
  Extracts points from a JSON file and returns them as a list of lists of tuples.
  Each element in the list  is another list that corresponds to a video frame and contains all the tracked points.
  '''
  
  with open(json_file, 'r') as f:
    data = json.load(f)
    
  # Extract points from the JSON data by iterating through each nested list and converting each point to a tuple
  points = {
    f"frame_{i}": [tuple(point) for line in frame for point in line]
    for i, frame in enumerate(data)
  }
  
  return points


def main():
  # --- Configuration ---
  video_source = 'videos/hpe_01/out1.mp4'  # Path to the video file
  json_file = "camera_data_with_Rvecs_2ndversion/Camera_config2/cam_1/dump/dump_20240412_221319.json"  # Path to the JSON file 
  video_dump = 'tests/dump_video.mp4'  # Path to the output video file
  
  # Check file existence
  if not os.path.exists(video_source):
    print(f"Error: Video file '{video_source}' does not exist.")
    return
  if not os.path.exists(json_file):
    print(f"Error: JSON file '{json_file}' does not exist.")
    return
  
  # --- Extract points from JSON ---
  points = extract_points_from_json(json_file)
  
  # --- Open video source ---
  cap = cv2.VideoCapture(video_source) # Open the video file
  fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height
  
  total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  total_json_frames = len(points)
  
  print("=" * 40)
  print(f"Metadata:")
  print(f"-> Video Frames: {total_video_frames}")
  print(f"-> JSON Frames:  {total_json_frames}")
  print("=" * 40)
  
  # --- Read the FIRST frame to use as a static background ---
  ret, first_frame = cap.read() 
  if not ret:
    print("Error: Could not read the first frame of the video.")
    cap.release()
    return

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec to have a .mp4 output
  out = cv2.VideoWriter(video_dump, fourcc, fps, (width, height))  # Create the output video  with the specified properties
  
  frame_index = 0
  # --- Process each frame of the JSON ---
  while frame_index < total_json_frames:
    
    # Use a copy of the first frame as the background for every JSON frame
    frame = first_frame.copy() 
    
    frame_points = points.get(f"frame_{frame_index}", [])  # Get the points of current frame
    
    # Draw each point on the frame
    for point in frame_points:
      x = int(point[0])  # Get the x-coordinate of the point
      y = int(point[1])  # Get the y-coordinate of the point
      
      # Ensure the point is within the frame boundaries
      if 0 <= x < width and 0 <= y < height: 
        cv2.circle(frame, (x, y), 8, (25, 0, 255), -1)  # Draw a green circle at the point
        
    out.write(frame)  # Write the modified frame to the output video
    frame_index += 1  # Increment the frame index
    
  # --- Release resources ---
  cap.release()  # Release the video capture object
  out.release()  # Release the video writer object
  
if __name__ == "__main__":
  main()  # Call the main function to execute the script