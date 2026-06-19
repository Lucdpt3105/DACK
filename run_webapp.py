"""
🚦 Vietnamese Traffic Sign Detection - Web App Launcher
=======================================================
Chỉ cần chạy file này, mở trình duyệt và sử dụng!

Cách dùng:
    python run_webapp.py
    
Sau đó mở: http://localhost:8000

Yêu cầu:
    pip install -r requirements-web.txt
"""

import os
import sys

# Thêm thư mục gốc vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from web_app.main import app
import uvicorn

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║   🚦 Vietnamese Traffic Sign Detection - WEB    ║
    ║   Nhận diện biển báo giao thông Việt Nam       ║
    ║                                                  ║
    ║   📍 http://localhost:8000                       ║
    ║                                                  ║
    ║   📸 Upload ảnh → Nhận diện ngay                ║
    ║   🎥 Upload video → Xử lý tự động               ║
    ║                                                  ║
    ║   🚫 Không cần cài đặt phức tạp!                ║
    ╚══════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)