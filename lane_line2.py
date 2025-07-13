import os
import argparse
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
import time

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
            'resize_factor': 1.0,
            'debug_mode': False,
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
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    def canny_detector(self, image):
        return cv2.Canny(
            image, 
            self.config['canny_low_threshold'], 
            self.config['canny_high_threshold']
        )
    
    def gaussian_smoothing(self, image):
        return cv2.GaussianBlur(
            image, 
            (self.config['kernel_size'], self.config['kernel_size']), 
            0
        )
    
    def region_selection(self, image):
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
        return cv2.HoughLinesP(
            image, 
            1,  # rho
            np.pi/180,  # theta
            self.config['hough_threshold'], 
            minLineLength=self.config['min_line_length'], 
            maxLineGap=self.config['max_line_gap']
        )
    
    def average_slope_intercept(self, lines):
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
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
    
    def lane_lines(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        
        # Làm mịn làn đường qua nhiều khung hình
        left_lane, right_lane = self.smooth_lanes(left_lane, right_lane)
        
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)
        return left_line, right_line
    
    def draw_lane_lines(self, image, lines):
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
    
    def process_frame(self, image):
        # Thay đổi kích thước ảnh nếu cần
        original_size = image.shape[:2]
        if self.config['resize_factor'] != 1.0:
            width = int(image.shape[1] * self.config['resize_factor'])
            height = int(image.shape[0] * self.config['resize_factor'])
            image = cv2.resize(image, (width, height))
        
        # Xử lý ảnh
        color_select = self.HSL_color_selection(image)
        gray = self.gray_scale(color_select)
        smooth = self.gaussian_smoothing(gray)
        edges = self.canny_detector(smooth)
        region = self.region_selection(edges)
        hough = self.hough_transform(region)
        result = self.draw_lane_lines(image, self.lane_lines(image, hough))
        
        # Lưu các ảnh debug nếu cần
        if self.config['debug_mode']:
            debug_images = {
                'color_select': color_select,
                'edges': cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB),
                'region': cv2.cvtColor(region, cv2.COLOR_GRAY2RGB) if len(region.shape) < 3 else region
            }
            # Trả về một từ điển chứa tất cả dữ liệu
            return {'result': result, 'debug': debug_images}
        
        # Thay đổi kích thước ảnh về kích thước ban đầu nếu cần
        if self.config['resize_factor'] != 1.0:
            result = cv2.resize(result, (original_size[1], original_size[0]))
            
        return result

def parse_arguments():
    parser = argparse.ArgumentParser(description='Phát hiện làn đường trong video')
    parser.add_argument('--input-dir', type=str, default='test_videos', help='Thư mục chứa video đầu vào')
    parser.add_argument('--output-dir', type=str, default='output', help='Thư mục lưu video kết quả')
    parser.add_argument('--debug', action='store_true', help='Bật chế độ debug')
    parser.add_argument('--resize', type=float, default=1.0, help='Hệ số thay đổi kích thước ảnh khi xử lý')
    parser.add_argument('--video', type=str, default=None, help='Chỉ xử lý video cụ thể, mặc định xử lý tất cả')
    return parser.parse_args()

def process_videos(args):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Thư mục debug nếu cần
    if args.debug and not os.path.exists(os.path.join(args.output_dir, 'debug')):
        os.makedirs(os.path.join(args.output_dir, 'debug'))
    
    # Cấu hình detector
    config = {
        'resize_factor': args.resize,
        'debug_mode': args.debug
    }
    detector = LaneDetector(config)
    
    # Danh sách video để xử lý
    if args.video:
        test_videos = [args.video]
    else:
        test_videos = [f for f in os.listdir(args.input_dir) if f.endswith('.mp4')]
    
    # Đo thời gian xử lý
    start_time = time.time()
    
    for test_video in test_videos:
        input_path = os.path.join(args.input_dir, test_video)
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(test_video)[0]}_output.mp4")
        
        print(f"Đang xử lý {input_path}...")
        video_clip = VideoFileClip(input_path)
        
        # Xử lý video
        processed_clip = video_clip.fl_image(detector.process_frame)
        
        # Lưu video kết quả
        processed_clip.write_videofile(output_path, audio=False, fps=video_clip.fps)
        print(f"Đã lưu video kết quả vào {output_path}")
    
    # Hiển thị thời gian xử lý
    total_time = time.time() - start_time
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây")

def main():
    args = parse_arguments()
    process_videos(args)

if __name__ == "__main__":
    main()