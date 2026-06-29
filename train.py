from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 1. Khởi tạo cấu trúc mạng YOLOv8 Nano (bản nhẹ nhất, phù hợp train máy cá nhân)
    model = YOLO('yolov8n.pt') 

    # 2. Định nghĩa đường dẫn tới file data.yaml vừa tải về
    # Vì Roboflow tải về thư mục 'heatmap-Batminton-1', ta trỏ trực tiếp vào đó
    yaml_path = os.path.join('.', 'heatmap-Batminton-1', 'data.yaml')

    # 3. Tiến hành huấn luyện mô hình
    model.train(
        data=yaml_path,     # Đường dẫn đến file data.yaml
        epochs=100,          # Lặp 50 lượt để AI học bối cảnh sân đấu
        imgsz=640,          # Kích thước ảnh chuẩn hóa đầu vào
        device='cpu',       # ĐỔI THÀNH device=0 nếu máy bạn có card rời NVIDIA để chạy nhanh gấp 10 lần
        workers=2           # Số luồng xử lý dữ liệu song song
    )