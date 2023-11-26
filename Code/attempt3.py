import cv2
import numpy as np
import os

# Object Dictionary - Maps object image to object name
objectDict = {
    'O1': 'Dawn Spry',
    'O2': 'Water Bottle',
    'O3': 'Spray Bottle',
    'O4': 'Candle',
    'O5': 'Book',
    'O6': 'Pliers',
    'O7': 'Paint Set',
    'O8': 'Strainer',
    'O9': 'Cup',
    'O10': 'Pot',
    'O11': 'Decoration',
    'O12': 'Lego Flower',
    'O13': 'Bowl',
    'O14': 'Axe',
    'O15': 'Ornament',
}

# Scene Dictionary - Maps scene to objects in scene
sceneDict = {
    'S1': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S2': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S3': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S4': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O10', 'O11', 'O14', 'O15'],
    'S5': ['O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O10', 'O12', 'O13', 'O14', 'O15'],
    # 'S6': ['O2', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S7': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S8': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S9': ['O5', 'O7', 'O8', 'O9', 'O10', 'O12', 'O13', 'O14'],
    # 'S10': ['O5', 'O7', 'O8', 'O9', 'O10', 'O12', 'O13'],
    # 'S11': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S12': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O10', 'O11', 'O13', 'O14', 'O15'],
    # 'S13': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S14': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O13', 'O14', 'O15'],
    # 'S15': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O11', 'O14', 'O15'],
    # 'S16': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S17': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S18': ['O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S19': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S20': ['O5', 'O7', 'O8', 'O9', 'O10', 'O12', 'O13', 'O14'],
    # 'S21': ['O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O10', 'O13', 'O14', 'O15'],
    # 'S22': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O11', 'O14', 'O15'],
    # 'S23': ['O1', 'O2', 'O3', 'O4', 'O6', 'O11', 'O15'],
    # 'S24': ['O1', 'O2', 'O3', 'O4', 'O15'],
    # 'S25': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    # 'S26': ['O5', 'O7', 'O8', 'O9', 'O10', 'O12', 'O13', 'O14'],
    #'S27': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    #'S28': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O11', 'O14', 'O15'],
    #'S29': ['O1', 'O2', 'O3', 'O4', 'O6', 'O7', 'O8', 'O11', 'O14', 'O15'],
    #'S30': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    #'S31': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    #'S32': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
}


# Directory paths
object_images_path = "Objects"
scene_images_path = "Scenes"
detected_objects_path = "Detected_Objects"
keypoints_path = "Keypoints"
matches_path = "Matches"

# Ensure output directories exist
for path in [detected_objects_path, keypoints_path, matches_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Function to draw bounding boxes and annotate objects on scene images
def draw_boxes_and_annotate(scene_img, object_name, keypoints, matches):
    object_img_path = os.path.join(object_images_path, f"{object_name}.png")
    object_img = cv2.imread(object_img_path)

    # Draw bounding boxes around the matched keypoints
    scene_with_boxes = scene_img.copy()
    for match in matches:
        img1_idx = match.queryIdx
        (x, y) = keypoints[img1_idx].pt
        cv2.rectangle(scene_with_boxes, (int(x - 5), int(y - 5)), (int(x + 5), int(y + 5)), (0, 255, 0), 2)
        cv2.putText(scene_with_boxes, object_name, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return scene_with_boxes

# Function to compute SIFT features and match keypoints
def compute_sift_matches(object_img, scene_img):
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(object_img, None)
    kp2, des2 = sift.detectAndCompute(scene_img, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw bounding boxes and annotate objects on the scene image
    scene_img_with_boxes = draw_boxes_and_annotate(scene_img, object_name, kp2, good_matches)

    # Save the result image
    output_path = os.path.join(detected_objects_path, f"{scene_num}_detected.jpg")
    cv2.imwrite(output_path, scene_img_with_boxes)

    return len(good_matches)

# Initialize variables for metrics
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

# Iterate through scenes
for scene_num, object_list in sceneDict.items():
    scene_img_path = os.path.join(scene_images_path, f"{scene_num}.jpg")
    scene_img = cv2.imread(scene_img_path, cv2.IMREAD_GRAYSCALE)

    # Iterate through objects in the scene
    for object_name in object_list:
        object_img_path = os.path.join(object_images_path, f"{object_name}.png")
        object_img = cv2.imread(object_img_path, cv2.IMREAD_GRAYSCALE)

        # Compute SIFT matches
        matches = compute_sift_matches(object_img, scene_img)

        # Check if the detected object matches the ground truth
        if matches > 0:
            if object_name in object_list:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if object_name in object_list:
                false_negatives += 1
            else:
                true_negatives += 1

# Calculate metrics
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)

# Display metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Accuracy: {accuracy:.2f}")