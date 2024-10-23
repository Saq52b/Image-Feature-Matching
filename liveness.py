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
        self.root.geometry("800x600") 

        self.saved_images = []  
        self.image_labels = [] 
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=30, pady=10)
        self.camera_button = tk.Button(self.root, text="Open Camera", command=self.start_camera)
        self.camera_button.pack(padx=2,pady=10)
        self.save_button = tk.Button(self.root, text="Capture Image", command=self.save_frame)
        self.save_button.pack(pady=10)
        self.feature_button = tk.Button(self.root, text="Feature Matching", command=self.feature_matching)
        self.feature_button.pack(pady=10)
        self.SmoothImg_button = tk.Button(self.root, text="Smooth Image", command=self.Smoot)
        self.SmoothImg_button.pack(pady=10)
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
            #-----------------display
        for i, matches in enumerate(all_matches):
            img_matches = cv2.drawMatches(self.saved_images[i], keypoints_list[i], self.saved_images[i+1], keypoints_list[i+1], matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_matches_rgb))
            match_label = tk.Label(self.root, image=img_tk)
            match_label.image = img_tk
            match_label.pack(pady=10)

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
                self.canvas.create_window(400 * col, 200 * row, anchor='nw', window=image_label)
                self.canvas.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
