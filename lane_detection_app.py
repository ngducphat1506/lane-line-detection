import streamlit as st
import numpy as np
import cv2
import os
import tempfile
from moviepy.editor import VideoFileClip
#thay đổi thử thôi
class LaneDetector:
    def __init__(self, config=None):
        # Cấu hình mặc định
        self.config = {
            'kernel_size': 5,
            'canny_low_threshold': 50,
            'canny_high_threshold': 150,
            'hough_threshold': 20,
            'min_line_length': 20,
            'max_line_gap': 300,
            'line_color': [255, 0, 0],
            'line_thickness': 12
        }
        
        # Cập nhật cấu hình từ tham số
        if config:
            self.config.update(config)
        
        # Dữ liệu để làm mịn làn đường
        self.left_line_history = []
        self.right_line_history = []
        self.history_size = 10
        
    def gray_scale(self, image):
        """Chuyển đổi ảnh sang thang độ xám"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def canny_detector(self, image):
        """Áp dụng Canny detector cho ảnh"""
        return cv2.Canny(
            image, 
            self.config['canny_low_threshold'], 
            self.config['canny_high_threshold']
        )
    
    def gaussian_smoothing(self, image, kernel_size=None):
        """Áp dụng Gaussian Blur cho ảnh"""
        if kernel_size is None:
            kernel_size = self.config['kernel_size']
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def region_selection(self, image):
        """Chọn vùng quan tâm trong ảnh"""
        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def hough_transform(self, image):
        """Áp dụng Hough Transform để tìm đường thẳng"""
        return cv2.HoughLinesP(
            image, 
            1,  # rho
            np.pi/180,  # theta
            self.config['hough_threshold'], 
            minLineLength=self.config['min_line_length'], 
            maxLineGap=self.config['max_line_gap']
        )
    
    def average_slope_intercept(self, lines):
        """Tìm các đường thẳng trung bình của làn đường bên trái và bên phải"""
        left_lines = []
        left_weights = []
        right_lines = []
        right_weights = []
        
        if lines is None:
            return None, None
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)
                length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
                    
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
        
        return left_lane, right_lane
    
    def smooth_lanes(self, left_lane, right_lane):
        """Làm mịn làn đường qua nhiều khung hình"""
        # Thêm làn đường mới vào lịch sử
        if left_lane is not None:
            self.left_line_history.append(left_lane)
            if len(self.left_line_history) > self.history_size:
                self.left_line_history.pop(0)
        
        if right_lane is not None:
            self.right_line_history.append(right_lane)
            if len(self.right_line_history) > self.history_size:
                self.right_line_history.pop(0)
        
        # Lấy giá trị trung bình từ lịch sử
        left_avg = np.mean(self.left_line_history, axis=0) if self.left_line_history else None
        right_avg = np.mean(self.right_line_history, axis=0) if self.right_line_history else None
        
        return left_avg, right_avg
    
    def pixel_points(self, y1, y2, line):
        """Chuyển đổi slope và intercept thành các điểm pixel"""
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    
    def lane_lines(self, image, lines):
        """Tạo các đường thẳng làn đường đầy đủ từ các điểm pixel"""
        left_lane, right_lane = self.average_slope_intercept(lines)
        
        # Làm mịn làn đường qua nhiều khung hình
        left_lane, right_lane = self.smooth_lanes(left_lane, right_lane)
        
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)
        return left_line, right_line
    
    def draw_lane_lines(self, image, lines):
        """Vẽ các đường làn trên ảnh đầu vào"""
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(
                    line_image, 
                    *line, 
                    self.config['line_color'], 
                    self.config['line_thickness']
                )
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
    
    def HSL_color_selection(self, image):
        """Áp dụng lựa chọn màu HSL để làm nổi bật đường kẻ làn đường trắng và vàng"""
        # Chuyển đổi sang HSL
        converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        
        # Mặt nạ màu trắng
        lower_threshold = np.uint8([0, 200, 0])
        upper_threshold = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
        
        # Mặt nạ màu vàng
        lower_threshold = np.uint8([10, 0, 100])
        upper_threshold = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
        
        # Kết hợp các mặt nạ
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image
    
    def process_frame(self, image, return_steps=False):
        """Xử lý khung hình để phát hiện làn đường"""
        # Xử lý ảnh
        color_select = self.HSL_color_selection(image)
        gray = self.gray_scale(color_select)
        smooth = self.gaussian_smoothing(gray)
        edges = self.canny_detector(smooth)
        region = self.region_selection(edges)
        hough = self.hough_transform(region)
        result = self.draw_lane_lines(image, self.lane_lines(image, hough))
        
        if return_steps:
            steps = {
                'original': image,
                'color_select': color_select,
                'edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) if len(edges.shape) < 3 else edges,
                'region': cv2.cvtColor(region, cv2.COLOR_GRAY2RGB) if len(region.shape) < 3 else region,
                'result': result
            }
            return steps
        
        return result

def process_video(uploaded_video, config=None, progress_callback=None):
    """Xử lý video với detector và trả về đường dẫn tới video kết quả"""
    if uploaded_video is None:
        return None
    
    # Lưu video tạm
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    input_filename = tfile.name
    
    # Tạo tên file đầu ra
    output_filename = f"{input_filename}_processed.mp4"
    
    # Tạo detector
    detector = LaneDetector(config)
    
    # Xử lý video
    try:
        video_clip = VideoFileClip(input_filename)
        
        # Định nghĩa wrapper để cập nhật tiến độ
        def process_frame_with_progress(frame, frame_index=None, total_frames=None):
            if progress_callback and frame_index is not None and total_frames is not None:
                progress = int(100 * frame_index / total_frames)
                progress_callback(progress)
            return detector.process_frame(frame)
        
        # Xử lý mỗi khung hình với tiến độ
        frames = list(video_clip.iter_frames())
        total_frames = len(frames)
        
        processed_frames = []
        for i, frame in enumerate(frames):
            processed_frame = process_frame_with_progress(frame, i, total_frames)
            processed_frames.append(processed_frame)
        
        # Tạo clip mới từ các khung hình đã xử lý
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        processed_clip = ImageSequenceClip(processed_frames, fps=video_clip.fps)
        
        # Lưu video kết quả
        processed_clip.write_videofile(output_filename, audio=False)
        
        # Đóng và xóa file tạm sau khi đã xử lý xong
        video_clip.close()
        try:
            if os.path.exists(input_filename):
                os.unlink(input_filename)
        except PermissionError:
            # Bỏ qua lỗi khi không thể xóa file do file đang được sử dụng
            st.warning("Không thể xóa file tạm. File sẽ được xóa khi ứng dụng đóng.")
            pass
            
        return output_filename
    
    except Exception as e:
        st.error(f"Lỗi khi xử lý video: {str(e)}")
        # Cố gắng xóa file tạm trong trường hợp có lỗi
        try:
            if os.path.exists(input_filename):
                os.unlink(input_filename)
        except:
            pass
        return None

def main():
    st.set_page_config(
        page_title="Phát hiện làn đường",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 Ứng dụng phát hiện làn đường")
    
    # Sidebar cho cấu hình
    st.sidebar.header("Cấu hình phát hiện làn đường")
    
    # CSS cho tooltip
    st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 250px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        top: -5px;
        left: 125%;
        margin-left: 0px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
    }
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 15px;
        right: 100%;
        margin-top: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: transparent #555 transparent transparent;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hàm tạo tooltip
    def create_tooltip(label, description):
        return f"""
        <div class="tooltip">{label} ℹ️
            <span class="tooltiptext">{description}</span>
        </div>
        """
    
    # Kích thước kernel Gaussian với tooltip
    st.sidebar.markdown(create_tooltip(
        "Kích thước kernel Gaussian",
        "Điều chỉnh độ mịn của hình ảnh. Giá trị cao hơn sẽ làm mờ hình ảnh nhiều hơn, giúp giảm nhiễu nhưng có thể làm mất chi tiết."
    ), unsafe_allow_html=True)
    kernel_size = st.sidebar.slider("", 3, 15, 5, 2, key="kernel_size")
    
    # Ngưỡng dưới Canny với tooltip
    st.sidebar.markdown(create_tooltip(
        "Ngưỡng dưới Canny",
        "Ngưỡng dưới cho thuật toán Canny edge detection. Các cạnh có gradient thấp hơn giá trị này sẽ bị loại bỏ."
    ), unsafe_allow_html=True)
    canny_low = st.sidebar.slider("", 30, 100, 50, 5, key="canny_low")
    
    # Ngưỡng trên Canny với tooltip
    st.sidebar.markdown(create_tooltip(
        "Ngưỡng trên Canny",
        "Ngưỡng trên cho thuật toán Canny edge detection. Các cạnh có gradient cao hơn giá trị này sẽ luôn được giữ lại."
    ), unsafe_allow_html=True)
    canny_high = st.sidebar.slider("", 100, 200, 150, 5, key="canny_high")
    
    # Ngưỡng Hough Transform với tooltip
    st.sidebar.markdown(create_tooltip(
        "Ngưỡng Hough Transform",
        "Số lượng điểm tối thiểu để xác định một đường thẳng. Giá trị cao hơn sẽ giảm số lượng đường thẳng phát hiện được nhưng tăng độ tin cậy."
    ), unsafe_allow_html=True)
    hough_threshold = st.sidebar.slider("", 10, 50, 20, 1, key="hough_threshold")
    
    # Độ dày đường kẻ với tooltip
    st.sidebar.markdown(create_tooltip(
        "Độ dày đường kẻ",
        "Độ dày của đường kẻ làn đường trên hình ảnh kết quả."
    ), unsafe_allow_html=True)
    line_thickness = st.sidebar.slider("", 5, 20, 12, 1, key="line_thickness")
    
    # Màu đường kẻ với tooltip
    st.sidebar.markdown(create_tooltip(
        "Màu đường kẻ",
        "Màu sắc của đường kẻ làn đường trên hình ảnh kết quả."
    ), unsafe_allow_html=True)
    color_options = {
        "Đỏ": [255, 0, 0],
        "Xanh lá": [0, 255, 0],
        "Xanh dương": [0, 0, 255],
        "Vàng": [255, 255, 0]
    }
    line_color_name = st.sidebar.selectbox("", list(color_options.keys()), index=0, key="line_color")
    line_color = color_options[line_color_name]
    
    # Cấu hình detector
    config = {
        'kernel_size': kernel_size,
        'canny_low_threshold': canny_low,
        'canny_high_threshold': canny_high,
        'hough_threshold': hough_threshold,
        'line_color': line_color,
        'line_thickness': line_thickness
    }
    
    # Tạo hai cột chính
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Đầu vào")
        
        # Upload video
        uploaded_video = st.file_uploader("Tải lên video", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            # Tạo nút xử lý
            process_button = st.button("Xử lý video")
            
            if process_button:
                # Hiển thị thanh tiến độ
                progress_bar = st.progress(0)
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                with st.spinner("Đang xử lý video..."):
                    # Xử lý video
                    processed_video_path = process_video(uploaded_video, config, update_progress)
                    
                    if processed_video_path:
                        # Lưu đường dẫn vào session state để có thể hiển thị ở cột bên phải
                        st.session_state.processed_video_path = processed_video_path
                        st.success("Xử lý video thành công!")
                    else:
                        st.error("Không thể xử lý video.")
        
        # Thử với ví dụ
        st.header("Hoặc thử với ví dụ")
        if st.button("Thử với video mẫu"):
            # Kiểm tra xem có thư mục test_videos không
            if os.path.exists("test_videos"):
                example_videos = [f for f in os.listdir("test_videos") if f.endswith(".mp4")]
                if example_videos:
                    # Sử dụng video đầu tiên làm mẫu
                    example_video_path = os.path.join("test_videos", example_videos[0])
                    with open(example_video_path, 'rb') as f:
                        example_video_bytes = f.read()
                    
                    st.video(example_video_bytes)
                    
                    with st.spinner("Đang xử lý video mẫu..."):
                        # Tạo file tạm từ video mẫu
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile.write(example_video_bytes)
                        tfile.close()
                        
                        # Tạo đối tượng file tương tự như uploaded_file
                        import io
                        from streamlit.runtime.uploaded_file_manager import UploadedFile
                        
                        class MockUploadedFile:
                            def __init__(self, path):
                                self.path = path
                            
                            def read(self):
                                with open(self.path, 'rb') as f:
                                    return f.read()
                        
                        mock_uploaded_file = MockUploadedFile(tfile.name)
                        
                        # Hiển thị thanh tiến độ
                        progress_bar = st.progress(0)
                        
                        def update_progress(progress):
                            progress_bar.progress(progress)
                        
                        # Xử lý video mẫu
                        processed_video_path = process_video(mock_uploaded_file, config, update_progress)
                        
                        # Xóa file tạm
                        os.unlink(tfile.name)
                        
                        if processed_video_path:
                            # Lưu đường dẫn vào session state
                            st.session_state.processed_video_path = processed_video_path
                            st.success("Xử lý video mẫu thành công!")
                        else:
                            st.error("Không thể xử lý video mẫu.")
                else:
                    st.error("Không tìm thấy video mẫu trong thư mục 'test_videos'")
            else:
                st.error("Không tìm thấy thư mục 'test_videos'")
    
    with col2:
        st.header("Kết quả")
        
        # Kiểm tra xem đã có video được xử lý chưa
        if 'processed_video_path' in st.session_state and os.path.exists(st.session_state.processed_video_path):
            # Hiển thị video đã xử lý
            with open(st.session_state.processed_video_path, 'rb') as f:
                st.video(f.read())
            
            # Tùy chọn tải xuống
            with open(st.session_state.processed_video_path, 'rb') as f:
                video_bytes = f.read()
            
            st.download_button(
                label="Tải xuống video đã xử lý",
                data=video_bytes,
                file_name="lane_detection_result.mp4",
                mime="video/mp4"
            )
        else:
            st.info("Video kết quả sẽ hiển thị tại đây sau khi xử lý.")
    
    # Thêm phần giới thiệu và hướng dẫn ở cuối trang
    st.markdown("""
    ## Giới thiệu
    
    Ứng dụng này sử dụng OpenCV để phát hiện làn đường trong video. Thuật toán đã được cải tiến để:
    
    - Phát hiện làn đường trắng và vàng thông qua lọc màu HSL
    - Xác định cạnh với Canny Edge Detector
    - Sử dụng Hough Transform để phát hiện đường thẳng
    - Tính toán và vẽ làn đường bên trái và bên phải
    
    ## Cách sử dụng
    
    1. Tải lên video của bạn hoặc sử dụng video mẫu
    2. Điều chỉnh các tham số ở thanh bên trái nếu cần
    3. Nhấn nút "Xử lý video"
    4. Xem kết quả và tải xuống video đã xử lý
    
    © 2025 - Ứng dụng phát hiện làn đường
    """)

if __name__ == "__main__":
    main()