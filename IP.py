import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def feature_matching(image1, image2):
    orb = cv2.ORB_create()

    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches, keypoints1, keypoints2

def calculate_disparity(matches, keypoints1, keypoints2):
    disparities = []
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        disparity = np.linalg.norm(np.array(pt1) - np.array(pt2))
        disparities.append(disparity)
    return disparities

def disparity_to_depth(disparity_map, focal_length, baseline):
    depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
    enhanced_depth = enhance_edges(depth_map)
    return depth_map, enhanced_depth

def smooth_depth_map(image_path):
    image = read_image(image_path)
    if image is None:
        return None
    smoothed_depth = cv2.bilateralFilter(image, 15, 75, 75)
    return smoothed_depth

def enhance_edges(depth_map):
    laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
    enhanced_depth = depth_map + laplacian
    return enhanced_depth

def read_image(image_path):
    if not os.path.isfile(image_path):
        print(f"File at path {image_path} does not exist.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at path {image_path} could not be loaded.")
        return None

    print(f"Image at path {image_path} loaded successfully.")
    return image

if __name__ == '__main__':
    image1 = read_image('D:/Saqib/IdentifyModifiedImage/Mycode/ImageProcessing/images/1.jpeg')
    image2 = read_image('D:/Saqib/IdentifyModifiedImage/Mycode/ImageProcessing/images/2.jpeg')
    
    if image1 is None or image2 is None:
        print("One or both images could not be loaded. Exiting.")
        exit()
    
    matches, keypoints1, keypoints2 = feature_matching(image1, image2)
    disparities = calculate_disparity(matches, keypoints1, keypoints2)
    print(f"Number of matches found: {len(matches)}")
    print("Disparities:")
    for idx, disparity in enumerate(disparities):
        print(f"Match {idx+1}: {disparity}")

    focal_length = 1000  # Replace with your camera's focal length in pixels
    baseline = 0.1  # Replace with your camera's baseline in meters
    
    depth_map, enhanced_depth = disparity_to_depth(np.array(disparities), focal_length, baseline)
    smooth_depth = smooth_depth_map('D:/Saqib/IdentifyModifiedImage/Mycode/ImageProcessing/images/1.jpeg')

    print("\nDepth Map (meters):")
    for idx, depth in enumerate(depth_map):
        print(f"Match {idx+1}: {depth}")

    # Display images using OpenCV
    cv2.imshow('Image 1', image1)
    cv2.imshow('Image 2', image2)
    if smooth_depth is not None:
        cv2.imshow('Smoothed Depth Map', smooth_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize depth map and enhanced depth map using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Depth Map')
    plt.plot(depth_map, 'o')
    
    plt.subplot(1, 2, 2)
    plt.title('Enhanced Depth Map')
    plt.plot(enhanced_depth, 'o')
    
    plt.show()
