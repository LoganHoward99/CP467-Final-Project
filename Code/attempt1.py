import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# -----------------------
# Paramters
# -----------------------
NUM_SCENES = 4
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

# --------------------------
# Ground Truth Dictionaries
# --------------------------

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

def evaluateResults(TP, FP, FN, TN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    return precision, recall, f1_score, accuracy

def detectObjects(sceneImgs, objectImgs, sceneDict, objectDict):
    sift = cv2.SIFT_create()
    results = []

    for sceneName, sceneObjects in sceneDict.items():
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        # Initialize an empty image for the current scene
        imgBoxes = sceneImgs[int(sceneName[1:]) - 1].copy()

        for objectName in objectDict.keys():
            objectImg = objectImgs[int(objectName[1:])-1]
            sceneImg = sceneImgs[int(sceneName[1:])-1]

            # Detect keypoints and compute descriptors
            kp1, des1 = sift.detectAndCompute(objectImg, None)
            kp2, des2 = sift.detectAndCompute(sceneImg, None)

            flann = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)
            matches = flann.knnMatch(des1, des2, k=2)   

            goodMatches = []
            for m, n in matches:
                if m.distance < GOOD_MATCH_TH * n.distance:
                    goodMatches.append(m)

            if objectName in sceneObjects:
                if goodMatches:
                    TP += 1
                else:
                    FN += 1
            else:
                if goodMatches:
                    FP += 1
                else:
                    TN += 1

            if goodMatches:
                # Draw bounding box and annotate the detected object
                for match in goodMatches:
                    cv2.circle(imgBoxes, (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])), 3, 255, -1)
                h, w = objectImg.shape[:2]
                cv2.rectangle(imgBoxes, (0, 0), (w, h), 255, 2)
                cv2.putText(imgBoxes, objectDict[objectName], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

                # Save images with keypoints
                cv2.imwrite(os.path.join("Keypoints", f"{sceneName}_{objectName}_keypoints.png"), imgBoxes)

                # Save images with matches
                img_matches = sceneImg.copy()
                img_matches = cv2.drawMatches(objectImg, kp1, sceneImg, kp2, goodMatches, img_matches,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(os.path.join("Matches", f"{sceneName}_{objectName}_matches.png"), img_matches)

        # Save the image for the current scene with detected objects
        cv2.imwrite(os.path.join("Detected_Objects", f"{sceneName}_detected.png"), imgBoxes)
        results.append((sceneName, TP, FP, FN, TN))

    # Calculate metrics for the complete dataset
    totalTP = sum(tp for _, tp, _, _, _ in results)
    totalFP = sum(fp for _, _, fp, _, _ in results)
    totalFN = sum(fn for _, _, _, fn, _ in results)
    totalTN = sum(tn for _, _, _, _, tn in results)

    precision, recall, f1_score, accuracy = evaluateResults(totalTP, totalFP, totalFN, totalTN)

    print("Results:")
    print("Scene\tTP\tFP\tFN\tTN")
    for scene_name, tp, fp, fn, tn in results:
        print(f"{scene_name}\t{tp}\t{fp}\t{fn}\t{tn}")

    print("\nMetrics for the complete dataset:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

def main():
    # Load in scene and object images
    sceneImgs = [cv2.imread(f"Scenes/S{i}.jpg", cv2.IMREAD_GRAYSCALE) for i in range (1, NUM_SCENES + 1)]
    objectImgs = [cv2.imread(f"Objects/O{j}.png", cv2.IMREAD_GRAYSCALE) for j in range (1, NUM_OBJECTS + 1)]

    detectObjects(sceneImgs, objectImgs, sceneDict, objectDict)

if __name__ == "__main__":
    main()