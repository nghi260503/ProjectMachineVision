import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
FRAME_SKIP = 5 
W_dst, H_dst = 410, 640 
HALF_H = H_dst // 2 

# Đọc ảnh sân tiêu chuẩn của bạn để làm nền
pitch_template = cv2.imread('san_cau_long_doc.jpg')
if pitch_template is None:
    print("Không tìm thấy file 'san_cau_long_doc.jpg', đang tạo nền xanh mặc định.")
    pitch_template = np.zeros((H_dst, W_dst, 3), dtype=np.uint8)
    pitch_template[:] = (20, 80, 65) 
else:
    pitch_template = cv2.resize(pitch_template, (W_dst, H_dst))

# Tạo MẶT NẠ CHỐNG TRÀN BIÊN KHUNG SÂN 
court_mask_dst = np.zeros((H_dst, W_dst), dtype=np.uint8)
pts_court_dst = np.array([[0, 0], [W_dst, 0], [W_dst, H_dst], [0, H_dst]], dtype=np.int32)
cv2.fillPoly(court_mask_dst, [pts_court_dst], 255) 

# Tọa độ gốc và Homography
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

# Khởi tạo ma trận nhiệt tích lũy vĩnh viễn (float32)
heatmap = np.zeros((H_dst, W_dst), dtype=np.float32) 

model = YOLO(r'D:\yolo heatmap\runs\detect\train-2\weights\best.pt')

# Vòng lặp xử lý video
current_frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    current_frame_index += 1
    if current_frame_index % FRAME_SKIP != 0: continue
        
    video_display = cv2.resize(frame, (video_display_w, int(frame.shape[0] * scale_ratio)))

    results = model(frame, verbose=False)[0]
    
    for box in results.boxes:
        if float(box.conf[0]) > 0.35:
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            
            # Tính toán vị trí tâm trên video để phân biệt hai cầu thủ
            y_center_video = (y_min + y_max) / 2 * scale_ratio
            box_height = y_max - y_min
            
            # HÀM DỜI TỌA ĐỘ CHÂN KHỚP CHUẨN CỦA BẠN
            x_foot = int((x_min + x_max) / 2)
            if y_center_video >= 320:
                y_foot = int(y_max - (box_height * 0.33)) # Áo trắng
            else:
                y_foot = int(y_max - (box_height * 0.08)) # Áo đỏ

            # Vẽ khung nhận diện lên video gốc
            x_min_disp, y_min_disp = int(x_min * scale_ratio), int(y_min * scale_ratio)
            x_max_disp, y_max_disp = int(x_max * scale_ratio), int(y_max * scale_ratio)
            x_foot_disp, y_foot_disp = int(x_foot * scale_ratio), int(y_foot * scale_ratio)
            cv2.rectangle(video_display, (x_min_disp, y_min_disp), (x_max_disp, y_max_disp), (0, 255, 0), 2)
            cv2.circle(video_display, (x_foot_disp, y_foot_disp), 4, (0, 0, 255), -1)

            if cv2.pointPolygonTest(court_polygon, (x_foot, y_foot), False) >= 0:
                transformed = cv2.perspectiveTransform(np.array([[[x_foot, y_foot]]], dtype=float), H)
                x_real, y_real = int(transformed[0][0][0]), int(transformed[0][0][1])
                if 0 <= x_real < W_dst and 0 <= y_real < H_dst:
                    if y_center_video < 355 and y_real >= HALF_H:
                        continue
                    if y_center_video >= 355 and y_real < HALF_H:
                        continue
                    heatmap[y_real, x_real] += 25

   #xử lí đồ họa màu sắc
    if np.max(heatmap) > 0:
        blur = cv2.GaussianBlur(heatmap, (45, 45), 0)
        blur_clipped = cv2.bitwise_and(blur, blur, mask=court_mask_dst)
        max_val = np.max(blur_clipped)
        heatmap_normalized = blur_clipped / max_val
        heatmap_gamma = np.power(heatmap_normalized, 0.5)
        img_norm = (heatmap_gamma * 255.0).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(img_norm, cv2.COLORMAP_JET)
        mask_linear = img_norm / 255.0
        mask_enhanced = np.power(mask_linear, 0.85) * 0.95 
        mask_enhanced = np.clip(mask_enhanced, 0.0, 1.0)    
        mask_3ch = cv2.merge([mask_enhanced, mask_enhanced, mask_enhanced])
        display_output = (heatmap_colored * mask_3ch + pitch_template * (1.0 - mask_3ch)).astype(np.uint8)
    else:
        display_output = pitch_template.copy()

    # Hiển thị video và bản đồ nhiệt
    cv2.imshow('1. Video Quet Toa Do', video_display)
    cv2.imshow('2. Ban Do Nhiet', display_output)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

if np.max(heatmap) > 0:
    cv2.imwrite("heatmap.png", display_output)
    print("Đã xuất heatmap!")