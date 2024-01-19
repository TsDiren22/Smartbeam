import cv2
import os
from glob import glob
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import concurrent.futures

# Constants
POSITION_THRESHOLD = 50
SIZE_THRESHOLD = 50

# Load model
directory = 'testing_binary_class_classification/'

model = load_model(directory+'testing_binary_class_classification_model_Adam.h5')

def extract_blinking_lights(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return []

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        w += 20
        h += 20
        x -= 10
        y -= 10
        rois.append((x, y, w, h))

    return rois

def are_similar_coordinates(coord1, coord2):
    x1, y1, w1, h1 = coord1
    x2, y2, w2, h2 = coord2

    position_diff = abs(x1 - x2) + abs(y1 - y2)
    size_diff = abs(w1 - w2) + abs(h1 - h2)

    return position_diff < POSITION_THRESHOLD and size_diff < SIZE_THRESHOLD

def main():
    data_folder = 'labeled_frames_fixed'
    framelist = [image_path for image_path in glob(os.path.join(data_folder, '*.jpg'))]

    merged_rois = []
    all_rois = []

    for image_path in framelist:
        rois = extract_blinking_lights(image_path)
        all_rois.extend(rois)

    if all_rois:
        for roi_coordinates in all_rois:
            add_to_merged = True

            for merged_roi_coordinates in merged_rois:
                if are_similar_coordinates(roi_coordinates, merged_roi_coordinates):
                    add_to_merged = False
                    break

            if add_to_merged:
                merged_rois.append(roi_coordinates)

        print(f'Number of blinking lights: {len(merged_rois)}')

        # for i, coordinates in enumerate(merged_rois):
        #     print(f'ROI {i}: {coordinates}')
        #     x, y, w, h = coordinates
        #     img = cv2.imread(framelist[0])
        #     roi = img[y:y+h, x:x+w]
        #     cv2.imshow(f'ROI {i}', roi)
    else:
        print("No ROIs found.")

    # Start timer
    timer = cv2.getTickCount()

    def process_roi(roi):
        x, y, w, h = roi
        roi_processed = frame[y:y+h, x:x+w]
        roi_processed = cv2.resize(roi_processed, (30, 30), interpolation=cv2.INTER_LINEAR)
        roi_processed = cv2.convertScaleAbs(roi_processed)
        roi_processed = np.array(roi_processed)
        roi_processed = roi_processed / 255.0
        roi_processed = np.expand_dims(roi_processed, axis=0)
        return roi_processed
    
    # Open the video file
    video_path = './vidscapstone/vid9.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    rois = merged_rois

    # Loop through frames
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process ROIs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            rois_processed = list(executor.map(process_roi, rois))

        rois_processed = np.vstack(rois_processed)  # Stack ROIs to create a batch

        pred = model.predict(rois_processed)

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    # Calculate FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    print(f"FPS: {fps}")

    # Calculate total time
    total_time = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
    print(f"Total time: {total_time} seconds")

if __name__ == "__main__":
    main()
