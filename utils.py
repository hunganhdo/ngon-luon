import av
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import numpy as np
import os
from class_names import class_names

# --- 1. LOAD MODEL & CSS ---
@st.cache_resource
def load_model():
    # Đảm bảo bạn có file yolov8n.pt trong thư mục model
    return YOLO("./model/yolov8n.pt")

def styling_css():
    # Load CSS nếu có
    if os.path.exists('./assets/css/general-style.css'):
        with open('./assets/css/general-style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- 2. HÀM HIỂN THỊ KẾT QUẢ (Dùng Streamlit Native - Không lỗi HTML) ---
def display_results(results, container_placeholder):
    # Xóa nội dung cũ trong khung chứa
    container = container_placeholder.container()
    
    with container:
        st.divider()
        st.subheader("🥗 Kết quả phân tích")
        
        total_calories = 0
        total_fat = 0
        found_any = False
        
        # Duyệt qua các kết quả
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                
                # Bỏ qua nếu ID lạ không có trong danh sách
                if class_id >= len(class_names): continue
                
                info = class_names[class_id]
                name = info["name"]
                conf = int(box.conf[0].item() * 100)
                nutri = info["nutrition"]
                serving = info["serving_type"]
                
                found_any = True
                total_calories += nutri.get('Calories', 0)
                total_fat += nutri.get('Fat', 0)
                
                # --- SỬA LỖI Ở ĐÂY: Dùng st.expander và st.metric thay vì HTML ---
                with st.expander(f"🔹 {name} (Độ tin cậy: {conf}%)", expanded=True):
                    st.caption(f"📏 Khẩu phần: {serving}")
                    
                    # Chia thành 4 cột để hiển thị chỉ số đẹp mắt
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔥 Calo", f"{nutri.get('Calories', 0)}")
                    c2.metric("🥩 Chất béo", f"{nutri.get('Fat', 0)}g")
                    c3.metric("🍬 Đường", f"{nutri.get('Sugar', 0)}g")
                    c4.metric("🧂 Muối", f"{nutri.get('Salt', 0)}g")

        # Hiển thị tổng kết
        if found_any:
            st.success(f"📊 **TỔNG KẾT:** Bữa ăn này khoảng **{total_calories} kcal** và **{total_fat}g chất béo**.")
        else:
            st.warning("⚠️ Không nhận diện được món ăn nào trong danh sách dữ liệu.")

# --- 3. CHỨC NĂNG: ẢNH ---
def detect_image(conf, uploaded_file, model):
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    
    if st.button("🔍 Phân tích ngay"):
        with st.spinner("Đang xử lý AI..."):
            results = model.predict(image, conf=conf)
            res_plotted = results[0].plot()
            
            # Chuyển màu BGR -> RGB để hiển thị đúng
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            with col2:
                st.image(res_image, caption="Kết quả nhận diện", use_container_width=True)
            
            # Gọi hàm hiển thị kết quả mới
            display_results(results, st.empty())

# --- 4. CHỨC NĂNG: VIDEO ---
def detect_video(conf, uploaded_file, model):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    stop_btn = st.button("⏹️ Dừng video")
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=conf)
        res_plotted = results[0].plot()
        
        # Hiển thị video realtime
        st_frame.image(res_plotted, channels="BGR", use_container_width=True)
    
    cap.release()

# --- 5. CHỨC NĂNG: WEBCAM ---
class VideoTransformer(VideoProcessorBase):
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(img, conf=self.conf)
        img_plotted = results[0].plot()
        return av.VideoFrame.from_ndarray(img_plotted, format="bgr24")

def detect_webcam(conf, model):
    webrtc_streamer(
        key="food-detection",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=lambda: VideoTransformer(conf, model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- 6. CHỨC NĂNG: IP CAMERA ---
def detect_camera(conf, model, address):
    cap = cv2.VideoCapture(address)
    st_frame = st.empty()
    stop_btn = st.button("Ngắt kết nối")
    
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể kết nối tới Camera IP.")
            break
            
        results = model.predict(frame, conf=conf)
        res_plotted = results[0].plot()
        st_frame.image(res_plotted, channels="BGR", use_container_width=True)
    
    cap.release()