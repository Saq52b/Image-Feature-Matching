import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk 
import matplotlib.pyplot as plt
import requests
import imutils

url = "http://192.168.0.89:8080/shot.jpg"

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Application")
        self.root.geometry("800x600")  # Set the window size
        self.saved_images = []  # List to store saved images
        self.image_labels = []  # List to store labels for saved images

        # Create label for camera frame
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=2, pady=3)

        # Create buttons
        self.camera_button = tk.Button(self.root, text="Open Camera", command=self.start_camera)
        self.camera_button.pack(side=tk.LEFT,padx=10, pady=4)
        self.save_button = tk.Button(self.root, text="Save Image", command=self.save_frame)
        self.save_button.pack(side=tk.LEFT,padx=10, pady=4)
        self.feature_button = tk.Button(self.root, text="Feature Matching", command=self.feature_matching)
        self.feature_button.pack(side=tk.LEFT,padx=10, pady=4)
        self.SmoothImg_button = tk.Button(self.root, text="Smooth Image", command=self.smooth_depth_map)
        self.SmoothImg_button.pack(side=tk.LEFT,padx=10, pady=4)
        self.canvas = tk.Canvas(self.root, width=800, height=400)
        self.canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        self.scrollbar = tk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.stop_camera = False 

    def start_camera(self):
        self.stop_camera = False
        self.update_frame()

    def update_frame(self):
        if not self.stop_camera:
            img_resp = requests.get(url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)
            img = imutils.resize(img, width=300, height=500)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.image_label.img_tk = img_tk
            self.image_label.config(image=img_tk)
            self.img = img
            self.root.after(10, self.update_frame)

    def feature_matching(self):
        self.stop_camera = True
        if len(self.saved_images) < 2:
            print("Need at least two images for feature matching")
            return
    
        orb = cv2.ORB_create()
        
        keypoints_list = []
        descriptors_list = []
        
        for img in self.saved_images:
            keypoints, descriptors = orb.detectAndCompute(img, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        all_matches = []
        
        for i in range(len(self.saved_images) - 1):
            matches = bf.match(descriptors_list[i], descriptors_list[i+1])
            matches = sorted(matches, key=lambda x: x.distance)
            all_matches.append(matches)
        
        # Display matched images using Matplotlib
        plt.figure(figsize=(15, 10))
        for i, matches in enumerate(all_matches):
            img_matches = cv2.drawMatches(self.saved_images[i], keypoints_list[i], self.saved_images[i+1], keypoints_list[i+1], matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
            
            plt.subplot(len(all_matches), 1, i + 1)
            plt.imshow(img_matches_rgb)
            plt.title(f'Matched Features between Image {i} and Image {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
        self.calculate_disparity(all_matches, keypoints_list)
        
    def calculate_disparity(self, all_matches, keypoints_list):
        disparities = []
        for i in range(len(all_matches)):
            matches = all_matches[i]
            keypoints1 = keypoints_list[i]
            keypoints2 = keypoints_list[i+1]
            disparity_for_image_pair = []
            for match in matches:
                pt1 = keypoints1[match.queryIdx].pt
                pt2 = keypoints2[match.trainIdx].pt
                disparity = np.linalg.norm(np.array(pt1) - np.array(pt2))
                disparity_for_image_pair.append(disparity)
            
            disparities.append(disparity_for_image_pair)
        
        focal_length = 1000  # Replace with your camera's focal length in pixels
        baseline = 0.1 
        self.disparity_to_depth(np.array(disparities), focal_length, baseline)

    def disparity_to_depth(self, disparity_map, focal_length, baseline):
        depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
        enhanced_depth = self.enhance_edges(depth_map)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Depth Map')
        plt.imshow(depth_map, cmap='jet')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title('Enhanced Depth Map')
        plt.imshow(enhanced_depth, cmap='jet')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    def enhance_edges(self, depth_map):
        laplacian = cv2.Laplacian(depth_map, cv2.CV_64F)
        enhanced_depth = depth_map + laplacian
        return enhanced_depth
    
    def smooth_depth_map(self):
        if len(self.saved_images) < 2:
            print("Need at least two images for feature matching")
            return
        
        plt.figure(figsize=(15, 5))
        for i, img in enumerate(self.saved_images, 1):
            smoothed_depth = cv2.bilateralFilter(img, d=15, sigmaColor=75, sigmaSpace=75)
            plt.subplot(1, len(self.saved_images), i)
            plt.title(f'Smooth Image {i}')
            plt.imshow(smoothed_depth, cmap='gray')  # Display the smoothed image in grayscale
            
        plt.tight_layout()
        plt.show()

    def save_frame(self):
        if self.img is not None:
            self.saved_images.append(self.img)
            self.canvas.delete("all")
            for i, saved_image in enumerate(self.saved_images):
                saved_image_rgb = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(saved_image_rgb))
                image_label = tk.Label(self.root, image=img_tk)
                image_label.image = img_tk  
                self.image_labels.append(image_label)
                row = i // 2
                col = i % 2
                self.canvas.create_window(200 * col, 100 * row, anchor='nw', window=image_label)
                self.canvas.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
