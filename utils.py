import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from class_names import class_names

# --- 1. LOAD MODEL (Chạy YOLOv8n) ---
@st.cache_resource
def load_model():
    # Đảm bảo bạn đã có file yolov8n.pt trong thư mục model
    return YOLO("./model/yolov8n.pt")

# --- 2. HÀM LẤY THÔNG TIN DINH DƯỠNG ---
def get_nutrition_info(class_id):
    # Kiểm tra ID có nằm trong danh sách dữ liệu không
    if class_id < len(class_names):
        return class_names[class_id]
    return None

# --- 3. HÀM XỬ LÝ CHÍNH (NHẬN DIỆN ẢNH) ---
def detect_image(conf, uploaded_file, model):
    # Đọc ảnh từ file upload
    image = Image.open(uploaded_file)
    
    # Chia giao diện thành 2 cột
    col1, col2 = st.columns(2)
    
    # Cột 1: Hiển thị ảnh gốc
    with col1:
        st.image(image, caption="Ảnh gốc", use_column_width=True)
    
    # Nút bấm để bắt đầu nhận diện
    if st.button("🔍 Phân tích Dinh Dưỡng"):
        with st.spinner("Đang phân tích món ăn..."):
            # Gọi model YOLOv8 để dự đoán
            results = model.predict(image, conf=conf)
            
            # Lấy kết quả vẽ bounding box (trả về mảng NumPy BGR)
            res_plotted = results[0].plot()
            
            # Chuyển đổi màu từ BGR (OpenCV) sang RGB (Pillow) để hiển thị đúng màu
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            # Cột 2: Hiển thị ảnh kết quả
            with col2:
                st.image(res_image, caption="Kết quả AI nhận diện", use_column_width=True)
            
            # --- PHẦN HIỂN THỊ THÔNG TIN DINH DƯỠNG ---
            st.divider()
            st.subheader("📊 Bảng Dinh Dưỡng (Ước tính)")
            
            found_any = False
            
            # Duyệt qua từng vật thể model tìm thấy
            for box in results[0].boxes:
                # Lấy ID của vật thể (ví dụ: 0, 1, 2...)
                class_id = int(box.cls[0].item())
                
                # Tìm thông tin trong file class_names.py
                food_info = get_nutrition_info(class_id)
                
                if food_info:
                    found_any = True
                    name = food_info['name']
                    serving = food_info['serving_type']
                    nutri = food_info['nutrition']
                    
                    # Tạo hộp thông tin chi tiết (Expander)
                    with st.expander(f"🍲 {name} ({serving})", expanded=True):
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Calories", f"{nutri.get('Calories', 0)} kcal")
                        c2.metric("Fat", f"{nutri.get('Fat', 0)}g")
                        c3.metric("Carbs", f"{nutri.get('Sugar', 0)}g")
                        c4.metric("Salt", f"{nutri.get('Salt', 0)}g")
                        c5.metric("Saturates", f"{nutri.get('Saturates', 0)}g")
            
            # Thông báo nếu không khớp dữ liệu
            if not found_any:
                st.warning("⚠️ Đã nhận diện được vật thể nhưng chưa có thông tin dinh dưỡng tương ứng trong dữ liệu.")