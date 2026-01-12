import streamlit as st
# Bỏ thư viện navbar đi để tránh lỗi
from utils import detect_image, detect_video, detect_webcam, detect_camera, load_model, styling_css

# 1. Cấu hình trang
st.set_page_config(
    page_title="FoodDetector Pro",
    page_icon="🍲",
    layout="wide"
)

# 2. Load CSS & Model
try:
    styling_css()
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Lỗi khởi động: {e}")
    st.stop()

# 3. Tạo Menu điều hướng (Dùng Sidebar chuẩn của Streamlit cho ổn định)
with st.sidebar:
    st.title("🍲 FoodDetector")
    selected_page = st.radio("Đi tới:", ["Trang chủ", "Giới thiệu", "Mã nguồn"])
    st.markdown("---")
    
    # Cài đặt độ tin cậy nằm luôn ở đây
    st.header("⚙️ Cài đặt")
    confidence = st.slider("Độ tin cậy (Confidence)", 10, 100, 40) / 100

# 4. Giao diện trang CHỦ
if selected_page == "Trang chủ":
    # --- ĐOẠN NÀY ĐÃ ĐƯỢC THAY ĐỔI ---
    # Dòng cũ: st.title("🕵️ Nhận diện & Tính Calo Món Ăn") -> Xóa đi hoặc thêm dấu # đằng trước
    
    # Dòng mới: Hiển thị banner
    try:
        st.image("welcome.png", use_container_width=True) 
    except:
        st.error("Lỗi: Không tìm thấy file welcome.png. Hãy chắc chắn bạn đã copy ảnh vào thư mục dự án!")
    # ----------------------------------

    st.markdown("Chọn chế độ đầu vào bên dưới:")

    # 4 Tab chức năng
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Ảnh", "🎥 Video", "📷 Webcam", "📹 IP Camera"])

    with tab1: # Tab Ảnh
        st.subheader("Tải ảnh món ăn")
        uploaded_file = st.file_uploader("Chọn ảnh (jpg, png)...", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            detect_image(confidence, uploaded_file, model)

    with tab2: # Tab Video
        st.subheader("Tải video món ăn")
        uploaded_video = st.file_uploader("Chọn video (mp4, avi)...", type=['mp4', 'avi'])
        if uploaded_video:
            detect_video(confidence, uploaded_video, model)

    with tab3: # Tab Webcam
        st.subheader("Camera trực tiếp")
        st.info("Bấm START để bật camera")
        detect_webcam(confidence, model)

    with tab4: # Tab IP Camera
        st.subheader("Kết nối Camera IP")
        rtsp_url = st.text_input("Nhập địa chỉ RTSP:", placeholder="rtsp://admin:pass@192.168.1.x:554/...")
        if st.button("Kết nối Camera"):
            if rtsp_url:
                detect_camera(confidence, model, rtsp_url)
            else:
                st.warning("Vui lòng nhập địa chỉ RTSP")

# 5. Giao diện trang GIỚI THIỆU
elif selected_page == "Giới thiệu":
    st.header("ℹ️ Về dự án")
    st.info("""
    **FoodDetector** là ứng dụng AI giúp nhận diện các món ăn Việt Nam và tính toán dinh dưỡng.
    - **Công nghệ:** YOLOv8n, Streamlit, OpenCV
    - **Dữ liệu:** 67 món ăn Việt Nam
    """)

elif selected_page == "Mã nguồn":
    st.header("📂 Mã nguồn")
    st.write("Truy cập GitHub của dự án tại...")