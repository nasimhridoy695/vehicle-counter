import numpy as np
import cv2

from tqdm import tqdm

# Initialize global variables to store the coordinates
points = []

# Mouse callback function to store the coordinates of the points clicked
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw a circle at the point clicked
        cv2.circle(param, (x, y), 5, (255, 0, 0), -1)
        # Display the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(param, str((x, y)), (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Frame", param)
        
        # Check if two points are collected
        if len(points) == 2:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

min_width_react = 30
min_hieght_react = 30
offset = 6
counter = 0

frame_index = 0
# Open Video
cap = cv2.VideoCapture('video_trim_fix.mp4')


def is_valid_frame(fid):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    return ret, frame

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Frame count: ", frame_count)
frameIds = []
while len(frameIds) < 50:
    fid = int(frame_count * np.random.uniform())
    ret, frame = is_valid_frame(fid)
    if ret:
        frameIds.append(fid)
        print(f"Valid frame added with ID: {fid}")  # Print the frame ID
        # Display the frame
        # cv2.imshow('Selected Frame', frame)
        # cv2.waitKey(0)  # Display each frame for 500 ms
        # cv2.destroyWindow('Selected Frame')
# frameIds = frame_count * np.random.uniform(size=25)


def is_within_bounds(cx, cy, point1, point2):
    x_min = min(point1[0], point2[0])
    x_max = max(point1[0], point2[0])
    y_min = min(point1[1], point2[1])
    y_max = max(point1[1], point2[1])
    
    return x_min <= cx <= x_max and y_min <= cy <= y_max

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

detect = []
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis manually
# frames_array = np.array(frames)
# medianFrame = np.zeros_like(frames_array[0])

# Median calculation
# medianFrame = manual_median(frames)

frames_array = np.array(frames)
medianFrame = np.median(frames_array, axis=0).astype(np.uint8)
# Display median frame
cv2.imshow('Frame', medianFrame)
cv2.imwrite('median_frame_1.jpg', medianFrame)
# cv2.imread('median.jpg', medianFrame)
cv2.setMouseCallback('Frame', click_event, medianFrame)
cv2.waitKey(0)

# Ensure that we have two points
if len(points) != 2:
    print("Please select exactly two points.")
    exit()

# Extract the points
point1, point2 = points

# Calculate the line equation coefficients using numpy
# Line equation: Ax + By + C = 0
A = point2[1] - point1[1]
B = point1[0] - point2[0]
C = point2[0] * point1[1] - point1[0] * point2[1]

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert median frame to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayMedianFrame', grayMedianFrame)
# Loop over all frames
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
pbar = tqdm(total=frame_count, desc='Processing Frames', unit='frames')
# Output video writer


out_abs_diff = cv2.VideoWriter('abs_diff_video.mp4', fourcc, fps, (frame_width, frame_height), False)
out_thresholded = cv2.VideoWriter('thresholded_video.mp4', fourcc, fps, (frame_width, frame_height), False)
out_morphologyex = cv2.VideoWriter('morphologyex_video.mp4', fourcc, fps, (frame_width, frame_height), False)
out_dilated = cv2.VideoWriter('dilated_video.mp4', fourcc, fps, (frame_width, frame_height), False)
# out_center = cv2.VideoWriter('center_video.mp4', fourcc, fps, (frame_width, frame_height), False)
out_final = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

ret = True
while ret:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    # if frame_index % 5 != 0:
    #     frame_index += 1
    #     pbar.update(1)
    #     continue
    # Convert current frame to grayscale manually
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference manually
    abs_diff_frame = np.abs(gray_frame.astype(np.int16) - grayMedianFrame.astype(np.int16)).astype(np.uint8)
    cv2.imshow('abs_diff_frame', abs_diff_frame)
    out_abs_diff.write(abs_diff_frame)
    # Threshold to binarize manually
    _, binarized_frame = cv2.threshold(abs_diff_frame, 30, 255, cv2.THRESH_BINARY)
    cv2.imshow('threshold', binarized_frame)
    out_thresholded.write(binarized_frame)
    # binarized_frame = thresholding(abs_diff_frame)
    # Morphological operations to remove noise and reduce object size
    kernel = np.ones((5, 5), np.uint8)
    binarized_frame = cv2.morphologyEx(binarized_frame, cv2.MORPH_OPEN, kernel)
    cv2.imshow('morphologyEx', binarized_frame)
    out_morphologyex.write(binarized_frame)
    binarized_frame = cv2.dilate(binarized_frame, kernel  , iterations=1)
    cv2.imshow('dilate', binarized_frame)
    out_dilated.write(binarized_frame)
    
    # Find contours
    contours, _ = cv2.findContours(binarized_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

 
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_hieght_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)
        
        for (cx, cy) in detect:
            if is_within_bounds(cx, cy, point1, point2):
            # Calculate the perpendicular distance from the point to the line
                distance = abs(A * cx + B * cy + C) / np.sqrt(A**2 + B**2)
                if distance < offset:
                    counter += 1
                    print(distance)
                    detect.remove((cx, cy))
                    print("Vehicle Counter:" + str(counter))

    # out_center.write(frame)
    # cv2.putText(frame, "VEHICLE COUNTER:" + str(counter), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    text = "VEHICLE COUNTER:" + str(counter)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Set the text position
    text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right edge
    text_y = text_size[1] + 10  # 10 pixels from the top edge

    # Put the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
    # Draw line after all processing
    frame_height, frame_width = frame.shape[:2]
    cv2.line(frame, point1, point2, (255, 127, 0), 3)
    
    # Display image
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(20) & 0xFF == 27:  # Press 'Esc' to exit
        break
    out_final.write(frame)
    
cap.release()
out_abs_diff.release()
out_thresholded.release()
out_morphologyex.release()
out_dilated.release()
out_final.release()
cv2.destroyAllWindows()
