import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

def gray_scale(image):
    """
    Chuyển đổi ảnh sang thang độ xám
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def canny_detector(image, low_threshold=50, high_threshold=150):
    """
    Áp dụng Canny detector cho ảnh
    """
    return cv2.Canny(image, low_threshold, high_threshold)

def gaussian_smoothing(image, kernel_size=5):
    """
    Áp dụng Gaussian Blur cho ảnh
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def region_selection(image):
    """
    Chọn vùng quan tâm trong ảnh
    """
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

def hough_transform(image):
    """
    Áp dụng Hough Transform để tìm đường thẳng
    """
    rho = 1
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 300
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

def average_slope_intercept(lines):
    """
    Tìm các đường thẳng trung bình của làn đường bên trái và bên phải
    """
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

def pixel_points(y1, y2, line):
    """
    Chuyển đổi slope và intercept thành các điểm pixel
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    Tạo các đường thẳng làn đường đầy đủ từ các điểm pixel
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Vẽ các đường làn trên ảnh đầu vào
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def HSL_color_selection(image):
    """
    Áp dụng lựa chọn màu HSL để làm nổi bật đường kẻ làn đường trắng và vàng
    """
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

def frame_processor(image):
    """
    Xử lý khung hình để phát hiện làn đường
    """
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    return result

def main():
    # Đảm bảo thư mục đầu ra tồn tại
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Danh sách video để xử lý
    test_videos = ['solidWhiteRight.mp4', 'solidYellowLeft.mp4']
    
    for test_video in test_videos:
        input_path = os.path.join('test_videos', test_video)
        output_path = os.path.join('output', f"{os.path.splitext(test_video)[0]}_output.mp4")
        
        print(f"Xử lý {input_path}...")
        video_clip = VideoFileClip(input_path)
        processed_clip = video_clip.fl_image(frame_processor)
        
        # Lưu video kết quả với fps được chỉ định rõ ràng
        processed_clip.write_videofile(output_path, audio=False, fps=video_clip.fps)
        print(f"Đã lưu video kết quả vào {output_path}")

if __name__ == "__main__":
    main() 