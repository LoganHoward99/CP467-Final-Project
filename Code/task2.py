import cv2
import numpy as np

# Function to load and preprocess the scene and object images
def load_images(scene_count, object_count):
    scene_images = [cv2.imread(f'Scenes/S{i}.jpg') for i in range(1, scene_count + 1)]
    object_images = [cv2.imread(f'Objects/O{j}.png') for j in range(1, object_count + 1)]

    # Preprocess images (resize, grayscale, etc.) if necessary

    return scene_images, object_images

# Function to perform feature matching and find homography
def find_homography(scene_img1, scene_img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find key points and descriptors for both images
    kp1, des1 = sift.detectAndCompute(scene_img1, None)
    kp2, des2 = sift.detectAndCompute(scene_img2, None)

    # FLANN parameters
    flann_params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    # Match key points between the two images
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Get corresponding points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Use RANSAC to find the homography matrix
    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography_matrix

# Function to stitch scene images together
def stitch_images(scene_images):
    # Initialize the stitched image as the first scene image
    stitched_scene_image = scene_images[0]

    # Iterate through pairs of adjacent scene images
    for i in range(1, len(scene_images)):
        homography_matrix = find_homography(stitched_scene_image, scene_images[i])

        # Warp the second image onto the first using the homography
        warped_image = cv2.warpPerspective(scene_images[i], homography_matrix,
                                           (stitched_scene_image.shape[1] + scene_images[i].shape[1],
                                            stitched_scene_image.shape[0]))

        # Blend the warped image with the first image to avoid visible seams
        stitched_scene_image[:, :scene_images[i].shape[1]] = warped_image[:, :scene_images[i].shape[1]]

    return stitched_scene_image

# Function to find the location of each object in the stitched scene image
def find_object_locations(stitched_scene_image, object_images):
    object_locations = []

    objects_images = [cv2.resize(obj_img, (stitched_scene_image.shape[1], stitched_scene_image.shape[0])) for obj_img in object_images]


    # Iterate through object images
    for obj_img in object_images:
        # Check if the object image is smaller than the stitched scene image
        if obj_img.shape[0] <= stitched_scene_image.shape[0] and obj_img.shape[1] <= stitched_scene_image.shape[1]:
            # Use template matching
            result = cv2.matchTemplate(stitched_scene_image, obj_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Record the location of each object in the stitched scene image
            object_locations.append(max_loc)
        else:
            print(f"Warning: Object image size ({obj_img.shape}) is larger than the stitched scene image size ({stitched_scene_image.shape}). Skipping.")

    return object_locations

# Main script
def main():
    scene_count = 5  # Update with the actual number of scene images
    object_count = 15  # Update with the actual number of object images

    # Load scene and object images
    scene_images, object_images = load_images(scene_count, object_count)

    # Stitch scene images together
    stitched_scene_image = stitch_images(scene_images)

    # Find the location of each object in the stitched scene image
    object_locations = find_object_locations(stitched_scene_image, object_images)

    # Display or save the final stitched scene image with object locations
    for i, loc in enumerate(object_locations):
        cv2.rectangle(stitched_scene_image, loc, (loc[0] + object_images[i].shape[1], loc[1] + object_images[i].shape[0]), (0, 255, 0), 2)

    cv2.imwrite("Stitched_Scene.png", stitched_scene_image)

if __name__ == "__main__":
    main()