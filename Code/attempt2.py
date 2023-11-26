import cv2
import numpy as np
import os

# -----------------------
# Paramters
# -----------------------
NUM_SCENES = 5
NUM_OBJECTS = 15

# SIFT parameters
N_FEATURES = 0
N_OCTAVE_LAYER = 3
CONTRAST_TH = 0.04
EDGE_TH = 10
SIGMA = 1.6

# FLANN
INDEX_PARAMS = dict(algorithm=1, trees=5)
SEARCH_PARAMS = dict(checks=50)

# Other
GOOD_MATCH_TH = 0.75
MATCH_TH = 10

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


# Task 1: Object detection using SIFT
def detect_objects(scene_image, object_images):
    sift = cv2.SIFT_create()
    flann_params = dict(algorithm=1, trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})

    # Find keypoints and descriptors for the scene
    kp_scene, des_scene = sift.detectAndCompute(scene_image, None)

    results = {}
    for obj_name, obj_image in object_images.items():
        kp_obj, des_obj = sift.detectAndCompute(obj_image, None)

        # Use FLANN to find the best matches
        matches = flann.knnMatch(des_obj, des_scene, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Store the number of matches for each object in the results dictionary
        results[obj_name] = len(good_matches)

    return results, kp_scene, good_matches, kp_obj

# Task 2: Evaluate detection results
def evaluate_results(results, scene_name):
    # Get ground truth from the scene dictionary
    ground_truth = set(sceneDict[scene_name])
    
    # Convert results dictionary to a set of detected objects
    detected_objects = {obj for obj, count in results.items() if count > 0}

    # Calculate true positives, false positives, true negatives, and false negatives
    true_positives = len(detected_objects.intersection(ground_truth))
    false_positives = len(detected_objects.difference(ground_truth))
    true_negatives = len(set(objectDict.keys()).difference(detected_objects).difference(ground_truth))
    false_negatives = len(ground_truth.difference(detected_objects))

    # Calculate precision, recall, F1-score, and accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)

    # Display and return the results
    print(f"Results for Scene {scene_name}:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print("\n")

    return true_positives, false_positives, true_negatives, false_negatives, precision, recall, f1_score, accuracy

# Load scene and object images
scene_images = {f"S{i}": cv2.imread(f"Scenes/S{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range(1, 6)}
object_images = {f"O{j}": cv2.imread(f"Objects/O{j}.png", cv2.IMREAD_GRAYSCALE) for j in range(1, 16)}

# Create folders for output
import os

output_folders = ['Detected_Objects', 'Keypoints', 'Matches']
for folder in output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Task 1 and 2 for each scene
for scene_name, scene_image in scene_images.items():
    # Task 1: Object detection using SIFT
    results, kp_scene, good_matches, kp_obj = detect_objects(scene_image, object_images)

    # Task 2: Evaluate detection results
    evaluate_results(results, scene_name)

    # Save annotated scene image with detected objects
    annotated_image = scene_image.copy()

    for obj_name, count in results.items():
        if count > 0:
            # Retrieve keypoint locations for the detected object
            obj_kp_scene = np.float32([kp_scene[m.trainIdx].pt for m in good_matches if m.queryIdx == obj_name])
            
            if obj_kp_scene.shape[0] > 0:
                # Calculate bounding box for the detected object
                obj_bbox = cv2.boundingRect(obj_kp_scene)

                # Draw bounding box on the annotated image
                cv2.rectangle(annotated_image, (int(obj_bbox[0]), int(obj_bbox[1])),
                            (int(obj_bbox[0] + obj_bbox[2]), int(obj_bbox[1] + obj_bbox[3])), (255, 0, 0), 2)

                # Annotate object name near the bounding box
                cv2.putText(annotated_image, objectDict[obj_name], (int(obj_bbox[0]), int(obj_bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
    cv2.imwrite(f"Detected_Objects/{scene_name}_detected.jpg", annotated_image)

    # Save keypoints images for scene and objects
    scene_keypoints_image = cv2.drawKeypoints(scene_image, kp_scene, None)
    cv2.imwrite(f"Keypoints/{scene_name}_keypoints.jpg", scene_keypoints_image)

    for obj_name, obj_image in object_images.items():
        obj_keypoints_image = cv2.drawKeypoints(obj_image, kp_obj, None)
        cv2.imwrite(f"Keypoints/{obj_name}_keypoints.jpg", obj_keypoints_image)

    # Save matches images for scene and objects
    for obj_name, count in results.items():
        if count > 0:
            matches_image = cv2.drawMatches(obj_image, kp_obj, scene_image, kp_scene, good_matches, None,
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"Matches/{scene_name}_{obj_name}_matches.jpg", matches_image)