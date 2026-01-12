import streamlit as st
from utils import load_model, detect_image

# Cấu hình trang
st.set_page_config(
    page_title="ngon luon - FoodDetector",
    page_icon="🍲",
    layout="wide"
)

# Tiêu đề
st.title("🍲 Dự án: ngon luon (FoodDetector V8)")
st.write("Ứng dụng nhận diện món ăn và tính toán dinh dưỡng sử dụng YOLOv8n.")

# Load model (chỉ load 1 lần)
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Lỗi: Không tìm thấy model YOLOv8n. Hãy chắc chắn bạn đã tải file 'yolov8n.pt' vào thư mục 'model'. Chi tiết: {e}")
    st.stop()

# Sidebar cài đặt
with st.sidebar:
    st.header("⚙️ Cài đặt")
    confidence = st.slider("Độ tin cậy (Confidence)", 10, 100, 40) / 100
    st.info("Phiên bản: Python 3.10 | Model: YOLOv8n")

# Giao diện chính
tab1, tab2 = st.tabs(["📸 Tải ảnh lên", "ℹ️ Hướng dẫn"])

with tab1:
    st.subheader("Nhận diện qua hình ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh món ăn (jpg, png)...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        detect_image(confidence, uploaded_file, model)

with tab2:
    st.markdown("""
    ### Hướng dẫn sử dụng:
    1. Tải ảnh món ăn lên ở tab "Tải ảnh lên".
    2. Nhấn nút **Phân tích Dinh Dưỡng**.
    3. Xem kết quả nhận diện và thông tin dinh dưỡng bên dưới.
    """)