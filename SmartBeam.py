import tensorflow as tf
import cv2
import numpy as np
import contextlib
import io
from datetime import datetime

def extract_blinking_lights(img):

    if img is None:
        print(f"Error: Unable to load image from {img}")
    else:
        # Convert the image to grayscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to highlight the lights
        _, thresholded = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract ROIs based on contours
        rois = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            w = w + 50
            h = h + 50
            x = x - 25
            y = y - 25
            
            rois.append((x,y,w,h))
        
        return rois

def are_similar_coordinates(coord1, coord2):
    x1, y1, w1, h1 = coord1
    x2, y2, w2, h2 = coord2
    
    # Adjust these threshold values based on your requirements
    position_threshold = 50  # Adjust as needed
    size_threshold = 50  # Adjust as needed
    
    # Check if the coordinates are similar in terms of position and size
    position_diff = abs(x1 - x2) + abs(y1 - y2)
    size_diff = abs(w1 - w2) + abs(h1 - h2)
    
    return position_diff < position_threshold and size_diff < size_threshold
    
video_calibration = './vidscapstone/vid10.mp4'
video_live = './vidscapstone/vid11.mp4'
model = tf.keras.models.load_model('./testing_binary_class_classification/testing_binary_class_classification_model_Adam.h5')

merged_rois = []
all_rois = []

cap = cv2.VideoCapture(video_calibration)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the duration of the video in seconds
video_duration = total_frames / fps

print(f"Video Duration: {video_duration:.2f} seconds")

framelistVideo = []

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rois = extract_blinking_lights(img)
    for roi in rois:
        all_rois.append(roi)
    
    framelistVideo.append(img)

if all_rois is not None:
    for roi_coordinates in all_rois:
        add_to_merged = True

        for merged_roi_coordinates in merged_rois:
            if are_similar_coordinates(roi_coordinates, merged_roi_coordinates):
                add_to_merged = False
                break
        
        if add_to_merged:
            merged_rois.append(roi_coordinates)

rois_in_frames = []
for img in framelistVideo:
    on_off = []
    breaker = False
    for roi_coordinates in merged_rois:
        x, y, w, h = roi_coordinates
        w = w + 50
        h = h + 50
        x = x - 25
        y = y - 25
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract the ROI
        roi = img[y:y+h, x:x+w]

        # Convert the ROI to grayscale and apply thresholding
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to highlight the lights
        _, thresholded = cv2.threshold(grey, 180, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            on_off.append(0)
        else:
            on_off.append(1)
    rois_in_frames.append(on_off)

# Transpose the array using zip
cal_freq = []
transposed_array = list(zip(*rois_in_frames))
for i, separated_array in enumerate(transposed_array):
    filtered_tuple = [separated_array[0]] 

    for i in range(1, len(separated_array)):
        if separated_array[i] != separated_array[i - 1]:
            filtered_tuple.append(separated_array[i])
            
    frequency = filtered_tuple.count(1)/video_duration
    print(f"Frequency of light {i}: {frequency}")
    cal_freq.append(frequency)

print(f'Number of blinking lights: {len(merged_rois)}')

#print all coordinates
for i, coordinates in enumerate(merged_rois):
    print(coordinates)
    x, y, w, h = coordinates
    roi = img[y:y+h, x:x+w]

global previous_frame

# Function to process each ROI
def process_roi(roi):

    x, y, w, h = roi
    x = x - 50
    y = y - 50
    w = w + 100
    h = h + 100
    
    roi_processed = frame[y:y+h, x:x+w]
    roi_processed = cv2.resize(roi_processed, (30, 30), interpolation=cv2.INTER_LINEAR)
    roi_processed = cv2.convertScaleAbs(roi_processed)
    roi_processed = np.array(roi_processed)
    roi_processed = roi_processed / 255.0
    roi_processed = np.expand_dims(roi_processed, axis=0)  # Add batch size dimension

    roi_processed_prev = previous_frame[y:y+h, x:x+w]
    roi_processed_prev = cv2.resize(roi_processed_prev, (30, 30), interpolation=cv2.INTER_LINEAR)
    roi_processed_prev = cv2.convertScaleAbs(roi_processed_prev)
    roi_processed_prev = np.array(roi_processed_prev)
    roi_processed_prev = roi_processed_prev / 255.0
    roi_processed_prev = np.expand_dims(roi_processed_prev, axis=0)  # Add batch size dimension
    
    return np.concatenate((roi_processed, roi_processed_prev))

cap = cv2.VideoCapture(video_live)

if not cap.isOpened():
    print("Error opening video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second from the video

# Initialize roi_frequencies for each ROI
roi_frequencies = {i: [] for i in range(len(merged_rois))}
reset_interval = 90

batch_size = len(merged_rois) * 2
frame_count = 0
frame_total = 0

timer = cv2.getTickCount()

print("Predictions have been started.")
print(merged_rois)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    frame_total += 1

    # Draw the merged ROIs on the frame

    if frame_count % 2 != 0 and reset_interval != frame_count:
        previous_frame = frame.copy()
        continue

    # Process ROIs sequentially
    rois_processed = [process_roi(roi) for roi in merged_rois]
    
    rois_processed = np.vstack(rois_processed)  # Stack ROIs to create a batch

    # Make predictions on the batch of ROIs
    with contextlib.redirect_stdout(io.StringIO()):
        predictions = model.predict(rois_processed)
        
    predictions = np.where(predictions > 0.5, 1, 0)
    ordered_predictions_1 = []
    ordered_predictions_2 = []
    
    for i, prediction in enumerate(predictions):
        if i % 2 == 0:
            ordered_predictions_1.append(prediction)
        else:
            ordered_predictions_2.append(prediction)

    predictions = np.concatenate((ordered_predictions_1, ordered_predictions_2))

    # Distribute predictions to each ROI
    for i, prediction in enumerate(predictions):
        roi_index = i % len(merged_rois)
        if len(roi_frequencies[roi_index]) == 0 or roi_frequencies[roi_index][-1] != prediction[0]:
            roi_frequencies[roi_index].append(prediction[0])

    # Reset frequencies after a certain number of frames
    if frame_count % reset_interval == 0:
        seconds_passed = frame_count / fps
        print(f"Seconds passed: {seconds_passed:.2f}")
        # Calculate the average frequency for each ROI
        for i in range(len(merged_rois)):
            if len(roi_frequencies[i]) > 0:
                frequency = sum(roi_frequencies[i]) / seconds_passed
                print(f"Frequency of light {i}: {frequency:.2f} Hz")
                if frequency < cal_freq[i] - 0.7 or frequency > cal_freq[1] + 0.7:
                    print(f"Inconsistencies on light {i} detected! Please check the light.")
                    print(f"The calculated Frequency is {frequency:.2f} Hz, while the expected frequency is {cal_freq[i]:.2f} Hz.")
                    with open('logs.txt', 'a') as f:
                        f.write(f"Date & Time: {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')} \nWarning: Expected frequency of light {i}: {cal_freq[i]:.2f}\nActual frequency: {frequency:.2f} Hz.\n\n")
                        f.close()

        roi_frequencies = {i: [] for i in range(len(merged_rois))}
        frame_count = 0
        print("Predicting next batch of frames...")

# Calculate FPS
fps = frame_total / ((cv2.getTickCount() - timer) / cv2.getTickFrequency())
print(f"FPS: {fps}")

# Calculate total time
total_time = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
print(f"Total time: {total_time} seconds")
cap.release()

# Release the OpenCV window
cv2.destroyAllWindows()
