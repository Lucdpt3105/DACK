"""
🚦 Vietnamese Traffic Sign Detection - Main GUI
==============================================
Giao diện chính để chọn tính năng: Ảnh, Video, hoặc Webcam
"""

import tkinter as tk
from tkinter import ttk, filedialog
import subprocess
import sys
import os

class TrafficSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Vietnamese Traffic Sign Detection")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Header
        header = tk.Label(
            root, 
            text="🚦 Nhận Diện Biển Báo Giao Thông", 
            font=("Arial", 18, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=15
        )
        header.pack(fill=tk.X)
        
        # Content frame
        frame = tk.Frame(root, bg="white", padx=20, pady=30)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Label
        label = tk.Label(
            frame,
            text="Chọn tính năng:",
            font=("Arial", 14),
            bg="white",
            fg="#2c3e50"
        )
        label.pack(pady=10)
        
        # Button styles
        btn_style = {
            "font": ("Arial", 12),
            "height": 2,
            "cursor": "hand2",
            "bd": 0,
            "relief": "raised"
        }
        
        # Image button
        self.btn_image = tk.Button(
            frame,
            text="📷 Nhận Diện Ảnh",
            command=self.run_image,
            bg="#3498db",
            fg="white",
            **btn_style
        )
        self.btn_image.pack(fill=tk.X, pady=8)
        self.btn_image.bind("<Enter>", lambda e: self.btn_image.config(bg="#2980b9"))
        self.btn_image.bind("<Leave>", lambda e: self.btn_image.config(bg="#3498db"))
        
        # Video button
        self.btn_video = tk.Button(
            frame,
            text="🎥 Nhận Diện Video",
            command=self.run_video,
            bg="#e74c3c",
            fg="white",
            **btn_style
        )
        self.btn_video.pack(fill=tk.X, pady=8)
        self.btn_video.bind("<Enter>", lambda e: self.btn_video.config(bg="#c0392b"))
        self.btn_video.bind("<Leave>", lambda e: self.btn_video.config(bg="#e74c3c"))
        
        # Webcam button
        self.btn_webcam = tk.Button(
            frame,
            text="📹 Nhận Diện Webcam",
            command=self.run_webcam,
            bg="#27ae60",
            fg="white",
            **btn_style
        )
        self.btn_webcam.pack(fill=tk.X, pady=8)
        self.btn_webcam.bind("<Enter>", lambda e: self.btn_webcam.config(bg="#229954"))
        self.btn_webcam.bind("<Leave>", lambda e: self.btn_webcam.config(bg="#27ae60"))
    
    def run_image(self):
        """Chạy chương trình nhận diện ảnh"""
        # Chọn file ảnh từ folder bienbao
        file_path = filedialog.askopenfilename(
            initialdir="bienbao",
            title="Chọn ảnh để nhận diện",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            subprocess.Popen([sys.executable, "Detection/image.py", "--image_path", file_path])
    
    def run_video(self):
        """Chạy chương trình nhận diện video"""
        video_path = "Video Project.mp4"
        subprocess.Popen([sys.executable, "Detection/inference.py", "--model", "model/best.pt", "--source", video_path, "--imgsz", "1280"])
    
    def run_webcam(self):
        """Chạy chương trình nhận diện webcam"""
        subprocess.Popen([sys.executable, "Detection/webcam.py"])

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficSignGUI(root)
    root.mainloop()
