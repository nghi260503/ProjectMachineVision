import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
FRAME_SKIP = 5 
W_dst, H_dst = 410, 640 

# Tọa độ gốc và Homography (giữ nguyên để map vào sân)
pts_src_display = [[320, 226], [745, 227], [861, 516], [205, 516]]
video_display_w = 960

cap = cv2.VideoCapture('clip.MOV')
ret, frame_init = cap.read()
if not ret:
    print("Không thể đọc video.")
    exit()

video_orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
scale_ratio = video_display_w / video_orig_w
pts_src = np.array([[pt[0] / scale_ratio, pt[1] / scale_ratio] for pt in pts_src_display], dtype=float)
court_polygon = pts_src.astype(np.int32)
pts_dst = np.array([[0, 0], [W_dst, 0], [W_dst, H_dst], [0, H_dst]], dtype=float)
H, _ = cv2.findHomography(pts_src, pts_dst)

# Khởi tạo ma trận nhiệt duy nhất cho TẤT CẢ VĐV
heatmap = np.zeros((H_dst, W_dst), dtype=np.float32) 

model = YOLO(r'D:\yolo heatmap\runs\detect\train-2\weights\best.pt')

# ==============================================================================
# VÒNG LẶP XỬ LÝ VÀ HIỂN THỊ SONG SONG
# ==============================================================================
current_frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    current_frame_index += 1
    if current_frame_index % FRAME_SKIP != 0: continue
        
    # Resize khung hình video để hiển thị
    video_display = cv2.resize(frame, (video_display_w, int(frame.shape[0] * scale_ratio)))

    results = model(frame, verbose=False)[0]
    
    for box in results.boxes:
        if float(box.conf[0]) > 0.4:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            x_foot, y_foot = int((x_min + x_max) / 2), int(y_max)
            
            # Kiểm tra xem chân có trong sân không
            if cv2.pointPolygonTest(court_polygon, (x_foot, y_foot), False) >= 0:
                transformed = cv2.perspectiveTransform(np.array([[[x_foot, y_foot]]], dtype=float), H)
                x_real, y_real = int(transformed[0][0][0]), int(transformed[0][0][1])
                
                if 0 <= x_real < W_dst and 0 <= y_real < H_dst:
                    heatmap[y_real, x_real] += 1 # Cộng dồn chung vào một ma trận

                # Vẽ khung nhận diện và điểm chân lên VIDEO HIỂN THỊ
                x_min_disp, y_min_disp = int(x_min * scale_ratio), int(y_min * scale_ratio)
                x_max_disp, y_max_disp = int(x_max * scale_ratio), int(y_max * scale_ratio)
                x_foot_disp, y_foot_disp = int(x_foot * scale_ratio), int(y_foot * scale_ratio)
                cv2.rectangle(video_display, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 255, 0), 2)
                cv2.circle(video_display, (x_foot_disp, y_foot_disp), 4, (0, 0, 255), -1)

    # Xử lý heatmap
    if np.max(heatmap) > 0:
        blur = cv2.GaussianBlur(heatmap, (71, 71), 0)
        img_norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
    else:
        heatmap_colored = np.zeros((H_dst, W_dst, 3), dtype=np.uint8)

    # Hiển thị song song hai cửa sổ
    cv2.imshow('1. Video Quet Toa Do', video_display)
    cv2.imshow('2. Ban Do Nhiet', cv2.resize(heatmap_colored, (380, 593)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Lưu file kết quả
if np.max(heatmap) > 0:
    cv2.imwrite("heatmap_final.png", heatmap_colored)
    print("Đã lưu bản đồ nhiệt tổng hợp.")
else:
    print("Không có dữ liệu heatmap để lưu.")