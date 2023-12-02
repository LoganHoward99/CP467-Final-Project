import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# -----------------------
# Paramters
# -----------------------

# FLANN
FLANN_INDEX_KDTREE = 1
FLANN_INDEX_PARAMS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
FLANN_SEARCH_PARAMS = dict(checks=50)

# Ratio Test
MATCH_TH = 0.55

# Object Dictionary - Maps object image to object name
objectDict = {
    'O1': 'Pot',
    'O2': 'Pliers',
    'O3': 'Cup',
    'O4': 'Dawn Spray',
    'O5': 'Lego Flower',
    'O6': 'Strainer',
    'O7': 'Paint Set',
    'O8': 'Decoration',
    'O9': 'Axe',
    'O10': 'Bowl',
    'O11': 'Book',
    'O12': 'Water Bottle',
    'O13': 'Decoration',
    'O14': 'Candle',
    'O15': 'Spray Bottle',
}

# Scene Dictionary - Maps scene to objects in scene
sceneDict = {
    'S1': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S2': ['O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    'S3': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12'],
    'S4': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10'],
    'S5': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9'],
    'S6': ['O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9'],
    'S7': ['O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13'],
    'S8': ['O6', 'O7', 'O9', 'O11', 'O12', 'O13', 'O14'],
    'S9': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10'],
    'S10': ['O6', 'O7', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S11': ['O2', 'O3', 'O4', 'O5', 'O15'],
    'S12': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O13', 'O14', 'O15'],
    'S13': ['O1', 'O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O11', 'O13', 'O14', 'O15'],
    'S14': ['O1', 'O2', 'O4', 'O5', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S15': ['O1', 'O2', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S16': ['O1', 'O2', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    'S17': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O14'],
    'S18': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11'],
    'S19': ['O1', 'O6', 'O7', 'O8', 'O9', 'O10'],
    'S20': ['O1', 'O7', 'O8', 'O9'],
    'S21': ['O1', 'O7', 'O8'],
    'S22': ['O1', 'O2', 'O7', 'O8', 'O11', 'O14', 'O15'],
    'S23': ['O2', 'O4', 'O7', 'O8', 'O15'],
    'S24': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O15'],
    'S25': ['O2', 'O3', 'O4', 'O5', 'O15'],
    'S26': ['O3', 'O4'],
    'S27': ['O2', 'O3', 'O4', 'O5', 'O7', 'O8', 'O13', 'O14', 'O15'],
    'S28': ['O1', 'O2', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O11', 'O12', 'O13', 'O14', 'O15'],
    'S29': ['O1', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14'],
    'S30': ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10', 'O11', 'O12', 'O13', 'O14', 'O15']
}


def SIFT(img):
    # Initialize
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)

    return keypoints, descriptors

def FLANN(des1, des2):
    # Initialize 
    flann = cv2.FlannBasedMatcher(FLANN_INDEX_PARAMS, FLANN_SEARCH_PARAMS)

    matches = flann.knnMatch(des1, des2, k=2)

    return matches
 
def evaluteScene(sceneID, sceneImg, truthObjectList):
    # Initialize 
    TP, FP, TN, FN = 0, 0, 0, 0

    # List for identified objects
    identifiedObjects = []

    sceneKeyPoints, sceneDescriptors = SIFT(sceneImg)

    # Iterate through objects 
    for objectID, objectName in objectDict.items():
        # Read in object image
        objectImg = cv2.imread(f"Objects/{objectID}.png", cv2.IMREAD_GRAYSCALE)

        # Get object key points and descriptors
        objectKeyPoints, objectDescriptors = SIFT(objectImg)

        # Matches descriptors
        matches = FLANN(objectDescriptors, sceneDescriptors)

        # Apply ratio test
        goodMatches = []
        for m, n in matches:
            if m.distance < MATCH_TH * n.distance:
                goodMatches.append(m)

        # Evaluate
        if objectID in truthObjectList:
            if goodMatches:
                TP += 1
                identifiedObjects.append(objectID)
            else:
                FN += 1
        else:
            if goodMatches:
                FP += 1
                identifiedObjects.append(objectID)
            else:
                TN += 1
        
    return TP, FP, TN, FN, identifiedObjects

def evaluateMetrics(TP, FP, TN, FN):
    # Create y_true and y_pred with consistent lengths
    y_true = [1] * TP + [0] * FN + [0] * FP + [1] * TN
    y_pred = [1] * TP + [0] * FN + [1] * FP + [0] * TN  
    
    # Calculate
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy

def keypoints():
    # Iterate over objects to save individual objects with keypoints
    for objectID, objectName in objectDict.items():
        # Read in object image
        objectImg = cv2.imread(f"Objects/{objectID}.png", cv2.IMREAD_GRAYSCALE)

        objectKeypoints, objectDescriptors = SIFT(objectImg)

        # Draw keypoints on object image
        objectKeypointsImg = objectImg.copy()
        objectKeypointsImg = cv2.drawKeypoints(objectImg, objectKeypoints, objectKeypointsImg)
        cv2.imwrite(f"Keypoints/{objectID}_keypoints.png", objectKeypointsImg)

def annotateAndBox(sceneID, sceneImg, identifiedObjects):
    # Draw bounding boxes and keypoints on the scene image
    annotatedScene = sceneImg.copy()

    for objectID in identifiedObjects:
        # Read in object image
        objectImg = cv2.imread(f"Objects/{objectID}.png", cv2.IMREAD_GRAYSCALE)

        # Get keypoints and descriptors
        objectKeypoints, objectDescriptors = SIFT(objectImg)
        sceneKeypoints, sceneDescriptors = SIFT(sceneImg)

        # FLANN ratio test
        matches = FLANN(objectDescriptors, sceneDescriptors)

        # Apply ratio test
        goodMatches = []
        for m, n in matches:
            if m.distance < MATCH_TH * n.distance:
                goodMatches.append(m)

        if len(goodMatches) > 10:  # Adjust as needed
            # Extract matched keypoints
            objectPts = np.float32([objectKeypoints[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
            scenePts = np.float32([sceneKeypoints[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

            # Find Homography matrix
            M, mask = cv2.findHomography(objectPts, scenePts, cv2.RANSAC, 5.0)

            # Draw bounding box
            h, w = objectImg.shape[:2]
            object_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            scene_corners = cv2.perspectiveTransform(object_corners, M)

            # Find the bounding box for the transformed object
            x, y, w, h = cv2.boundingRect(scene_corners.reshape(-1, 2).astype(int))

            # Draw bounding box only for the identified object
            if objectID in identifiedObjects:
                annotatedScene = cv2.rectangle(annotatedScene, (x, y), (x + w, y + h), 255, 2)

                # Annotate with object name at the center of the bounding box
                cv2.putText(annotatedScene, objectDict[objectID], (x + w//2, y + h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2, cv2.LINE_AA)

    # Save scene with boxes and keypoints
    cv2.imwrite(f"Detected_Objects/{sceneID}_detected.png", annotatedScene)

    # Write scene keypoints
    keypointsScene = sceneImg.copy()
    keypointsScene = cv2.drawKeypoints(sceneImg, sceneKeypoints, keypointsScene)
    cv2.imwrite(f"Keypoints/{sceneID}_keypoints.png", keypointsScene)

def matches():
    # Iterate over objects to save individual objects with keypoints
    for objectID, objectName in objectDict.items():
        # Read in object image
        objectImg = cv2.imread(f"Objects/{objectID}.png", cv2.IMREAD_GRAYSCALE)

        objectKeypoints, objectDescriptors = SIFT(objectImg)

        # Iterate over scenes to save matches
        for sceneID, scene_objects in sceneDict.items():
            # Read in scene image
            sceneImg = cv2.imread(f"Scenes/{sceneID}.jpg", cv2.IMREAD_GRAYSCALE)

            sceneKeypoints, sceneDescriptors = SIFT(sceneImg)

            matches = FLANN(objectDescriptors, sceneDescriptors)

            goodMatches = []
            for m, n in matches:
                if m.distance < MATCH_TH * n.distance:
                    goodMatches.append(m)

            # Draw matches
            matchedImg = cv2.drawMatches(objectImg, objectKeypoints, sceneImg, sceneKeypoints, goodMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Save match image
            cv2.imwrite(f"Matches/{sceneID}_{objectID}_matches.png", matchedImg)       

def main():
    # Initilaize metrics
    allTP, allFP, allTN, allFN = 0, 0, 0, 0

    # Iterate through all scenes
    for sceneID, truthObjectList in sceneDict.items():
        # Read in scene image
        sceneImg = cv2.imread(f"Scenes/{sceneID}.jpg", cv2.IMREAD_GRAYSCALE)

        TP, FP, TN, FN, identifiedObjects = evaluteScene(sceneID, sceneImg, truthObjectList)

        # Print reuslts for current scene
        print(f"Scene ID: {sceneID}")
        print("Ground Truth")
        print(truthObjectList)
        print("Predicted")
        print(identifiedObjects)
        print(f"True Positives: {TP}")
        print(f"False Positives: {FP}")
        print(f"True Negatives: {TN}")
        print(f"False Negatives: {FN}")

        # Annotate object names in scene image                      <---- need to add boxes
        annotateAndBox(sceneID, sceneImg, identifiedObjects)

        allTP += TP
        allFP += FP
        allTN += TN
        allFN += FN

    # Evaluate metrics of the entire dataset and capture the results
    precision, recall, f1, accuracy = evaluateMetrics(allTP, allFP, allTN, allFN)

    # Print overall metrics
    print("Overall Metrics:")
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')    

    matches()
    keypoints()

if __name__ == "__main__":
    main()


