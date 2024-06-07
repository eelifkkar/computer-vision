# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 00:14:00 2023

@author: Elif KAR
20190203011
"""
import cv2 as cv
import numpy as np
import time

def lanes_detection(img):
    # Convert the image to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    

    # Apply Gaussian blur to the grayscale image for less details in image
    blurred_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # Apply Canny edge detection to the blurred image.
    # First Treshold value:30. If it is lower than this threshold value, the edge is not accepted.
    # Second Treshold value:100. If it is higher than this threshold value, it is considered an edge.
    # Sobel core size:3
    edges = cv.Canny(blurred_img, 30, 100, apertureSize=3)

    # Define the region of interest (ROI)
    height, width = img.shape[:2]
    roi_vertices = [(0, height), (width//2, height//1.75), (width, height)]
    roi_mask = np.zeros_like(edges)
    cv.fillPoly(roi_mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv.bitwise_and(edges, roi_mask)

    # Use Hough line detection to detect lines in the masked image
    lines = cv.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)

    # Filter the detected lines by slope
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope < -0.5:
            left_lines.append(line)
        elif slope > 0.5:
            right_lines.append(line)

    # Find the average line for the left and right lanes
    def average_line(lines):
        x1 = np.mean([line[0][0] for line in lines])
        y1 = np.mean([line[0][1] for line in lines])
        x2 = np.mean([line[0][2] for line in lines])
        y2 = np.mean([line[0][3] for line in lines])
        return [int(x1), int(y1), int(x2), int(y2)]

    left_line = average_line(left_lines) if len(left_lines) > 0 else [0, 0, 0, 0]
    right_line = average_line(right_lines) if len(right_lines) > 0 else [0, 0, 0, 0]

    # Draw the left and right lane lines on a black image
    lane_lines = np.zeros((height, width, 3), dtype=np.uint8)
    thickness = 6
    cv.line(lane_lines, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 255), thickness)
    cv.line(lane_lines, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 255), thickness)

    # Overlay the lane lines on the original image
    result = cv.addWeighted(img, 0.8, lane_lines, 1, 0)

    return result, left_line, right_line

def has_lane_changed(prev_line, current_line, slope_threshold=15):
    # Check if there is a significant difference in slopes
    prev_slope = (prev_line[3] - prev_line[1]) / (prev_line[2] - prev_line[0] + 1e-1)
    current_slope = (current_line[3] - current_line[1]) / (current_line[2] - current_line[0] + 1e-1)
    
    return abs(current_slope - prev_slope) > slope_threshold

def video_lanes():
    # Open the video file
    cap = cv.VideoCapture('video.mp4')

    # Get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    output_file = 'output.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')

    # Create the VideoWriter object
    out = cv.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))

    # Initialize variables to store previous lane information
    prev_left_line = [0, 0, 0, 0]
    prev_right_line = [0, 0, 0, 0]
    last_lane_change_time = 0
    
    # Loop through the frames
    while cap.isOpened():
        # Read a frame from the video file
        ret, frame = cap.read()
        if ret:
            
            # Detect lane lines in the current frame
            lane_lines, left_line, right_line = lanes_detection(frame)

            # Check if the lanes have changed significantly
            if has_lane_changed(prev_left_line, left_line) or has_lane_changed(prev_right_line, right_line):
               last_lane_change_time = time.time()  # Record the time of the lane change
                
            # Update previous lane information
            prev_left_line = left_line.copy()
            prev_right_line = right_line.copy()

            # Calculate the time elapsed since the last lane change
            time_elapsed_since_lane_change = time.time() - last_lane_change_time

            # Display the "Lane Change Detected!" text for 1.5 seconds after a lane change
            if time_elapsed_since_lane_change < 1.5:
                cv.putText(lane_lines, "Lane Change Detected!", (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
            
            # Write the processed frame to the output video file
            out.write(lane_lines)

            # Display the processed frame
            cv.imshow('Lane Lines Detection', lane_lines)

            # Check if the user has pressed 'c' to quit
            if cv.waitKey(1) == ord('c'):
                break
        else:
            break

    # Release the resources
    cap.release()
    out.release()
    cv.destroyAllWindows()

# Run the video_lanes function if this script is being run as the main program
if __name__ == '__main__':
    video_lanes()