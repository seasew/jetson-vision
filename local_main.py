from red_contour_grip import RedContoursPipeline

import config

import cv2
import time
import math
import numpy

def calculate_coords(u, v):
    
    # pulled from https://github.com/ligerbots/VisionServer, BallFinder2020

    cam_matrix = config.cameramtx


    x_prime = (u - cam_matrix[0, 2]) / cam_matrix[0, 0]
    y_prime = -(v - cam_matrix[1, 2]) / cam_matrix[1, 1]

    ax = math.atan2(x_prime, 1.0)
    ay = math.atan2(y_prime * math.cos(ax), 1.0)

    print(str(ax) + " vs " + str(math.atan(x_prime)))
    
    target_height = 21
    camera_height = 4
    tilt_angle = 0


    d = (target_height - camera_height) / math.tan(tilt_angle + ay)

    return math.degrees(ax), math.degrees(ay), d

def calc_contours(c):
    M = cv2.moments(c)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

    return cx,cy

cap = cv2.VideoCapture(1)

# Run the pipeline on the video stream
while True:

    _, frame = cap.read()

    if frame is None:
        continue

    # Undistort the frame
    temp = cv2.undistort(frame, config.cameramtx, config.dist, None, config.cameramtx)
    # TODO Crop the frame using roi
    frame = temp

    # Process frame
    visionPipeline = RedContoursPipeline() # LemonVisionGripPipeline()
    visionPipeline.process(frame)

    # Retrieve the blobs from the pipeline
    contours = visionPipeline.filter_contours_output # tuple of KeyPoint objects

    print(str(len(contours)) + " blobs detected")

    if len(contours) > 0:
        # Find largest contour
        largest_contour = []
        largest_contour_idx = 0 
        for i in range(len(contours)):
            if (len(contours[i]) > len(largest_contour)):
                largest_contour = contours[i]
                largest_contour_idx = i

        # Find centroid of contour
        x, y = calc_contours(largest_contour)
        x = int(x)
        y = int(y)
        
        cv2.line(frame, (x - config.center_line_length, y), (x + config.center_line_length, y), config.color, config.line_thickness)
        cv2.line(frame, (x, y - config.center_line_length), (x, y + config.center_line_length), config.color, config.line_thickness)

        temp = cv2.drawContours(frame, contours, largest_contour_idx, config.color, config.line_thickness, cv2.LINE_8, maxLevel=0)
        frame = temp

        # Calculate angle to target
        pitch_angle, yaw_angle, d = calculate_coords(x, y)

        temp = cv2.putText(frame, "(" + str(round(pitch_angle, 3)) + ", " + str(round(yaw_angle, 3)) + ", " + str(round(d, 3)) +  ")", (x + config.text_offset, y - config.text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.color, config.line_thickness)
        frame = temp
        
        print("x: " + str(x) + "; y: " + str(y))
        print("pitch: " + str(pitch_angle) + "; yaw: " + str(yaw_angle))
        print(config.cameramtx[0, 2])
        print(config.cameramtx[0][2])

    cv2.imshow('image', frame)
    cv2.imshow('mask', visionPipeline.hsv_threshold_output)

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
