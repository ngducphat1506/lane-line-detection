# Phát hiện làn đường sử dụng Python và OpenCV

Ứng dụng phát hiện làn đường với giao diện trực quan giúp phát hiện và theo dõi làn đường trong video. Dự án này kết hợp các kỹ thuật xử lý ảnh và thị giác máy tính thông qua thư viện OpenCV để nhận diện làn đường trắng và vàng trên mặt đường.

## Tính năng

- Phát hiện làn đường trắng và vàng thông qua lọc màu HSL
- Xác định cạnh với Canny Edge Detector
- Sử dụng Hough Transform để phát hiện đường thẳng
- Làm mịn làn đường qua nhiều khung hình
- Giao diện người dùng trực quan với Streamlit
- Tùy chỉnh các tham số thuật toán với tooltip giải thích
- Xử lý video đầu vào và tạo video kết quả
- Hỗ trợ tải lên video tùy chọn hoặc sử dụng video mẫu

## Cài đặt

### Yêu cầu

- Python 3.7 trở lên
- OpenCV
- NumPy (phiên bản 1.24.3 hoặc thấp hơn)
- Streamlit
- MoviePy

### Hướng dẫn cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Nếu bạn gặp vấn đề với NumPy 2.0, hãy hạ cấp xuống phiên bản 1.24.3:

pip install numpy==1.24.3

2. Chạy ứng dụng:

python -m streamlit run lane_detection_app.py

## Cách sử dụng

1. Sau khi khởi chạy ứng dụng, một cửa sổ trình duyệt sẽ mở ra với giao diện Streamlit
2. Tải lên video của bạn (hỗ trợ định dạng MP4, AVI, MOV) hoặc sử dụng video mẫu 
3. Điều chỉnh các tham số ở sidebar bên trái nếu cần:
   - Kích thước kernel Gaussian (điều chỉnh độ mịn của hình ảnh)
   - Ngưỡng Canny (phát hiện cạnh)
   - Ngưỡng Hough Transform (phát hiện đường thẳng)
   - Độ dày và màu sắc của đường kẻ làn đường
4. Nhấn nút "Xử lý video" để bắt đầu quá trình phân tích
5. Sau khi xử lý hoàn tất, video kết quả sẽ hiển thị ở phần bên phải
6. Bạn có thể tải xuống video kết quả bằng nút "Tải xuống video đã xử lý"

## Cấu trúc mã nguồn

- **lane_detection_app.py**: File chính chứa ứng dụng Streamlit và các hàm xử lý video
- **LaneDetector**: Lớp chính để phát hiện làn đường với các phương thức:
  - HSL_color_selection: Lọc màu để tách biệt làn đường trắng và vàng
  - Canny detector: Phát hiện cạnh trong hình ảnh
  - Hough Transform: Chuyển đổi các điểm cạnh thành đường thẳng
  - Smooth lanes: Làm mịn làn đường qua nhiều khung hình

## Cải tiến

Dự án này đã trải qua một số cải tiến:
- Thêm tính năng làm mịn làn đường qua nhiều khung hình
- Chuyển sang kiến trúc hướng đối tượng để dễ bảo trì
- Thêm giao diện người dùng Streamlit
- Thêm tooltip giải thích cho các tham số
- Tối ưu hóa xử lý video

 