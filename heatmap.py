import cv2
import numpy as np
import os
import sys
import math


video_path = "clip.MOV"
cascade_path = "cascade.xml"  
court_bg_path = "san_cau_long_doc.jpg"   

SODO_W, SODO_H = 410, 640
frame_skip = 2  
PROCESS_W, PROCESS_H = 960, 540

if not os.path.exists(cascade_path):
    print(f" LỖI: Không tìm thấy file '{cascade_path}'!")
    sys.exit()

player_cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ LỖI: Không mở được video '{video_path}'!")
    sys.exit()

heatmap_matrix = np.zeros((SODO_H, SODO_W), dtype=np.float32)

clicked_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            cv2.circle(img_display, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(img_display, f"Goc {len(clicked_points)}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow("Buoc 1: Click 4 goc tham dau", img_display)

ret, first_frame = cap.read()
if not ret: sys.exit()

first_frame = cv2.resize(first_frame, (PROCESS_W, PROCESS_H))
img_display = first_frame.copy()

print("\n" + "="*70)
print(" HƯỚNG DẪN CLICK CHUỘT (THỨ TỰ BẮT BUỘC):")
print("  1. Click góc TRÊN - TRÁI ")
print("  2. Click góc TRÊN - PHẢI ")
print("  3. Click góc DƯỚI - PHẢI ")
print("  4. Click góc DƯỚI - TRÁI ")
print(" Click xong đủ 4 điểm, bấm một phím bất kỳ trên bàn phím để CHẠY.")
print("="*70 + "\n")

cv2.namedWindow("Buoc 1: Click 4 goc tham dau")
cv2.setMouseCallback("Buoc 1: Click 4 goc tham dau", click_event)
cv2.imshow("Buoc 1: Click 4 goc tham dau", img_display)
cv2.waitKey(0)
cv2.destroyWindow("Buoc 1: Click 4 goc tham dau")

if len(clicked_points) != 4: sys.exit()

court_polygon = np.array(clicked_points, dtype=np.int32)

pts_src = np.array(clicked_points, dtype=np.float32)
pts_dst = np.array([[0, 0], [SODO_W, 0], [SODO_W, SODO_H], [0, SODO_H]], dtype=np.float32)
H = cv2.getPerspectiveTransform(pts_src, pts_dst)

mid_y = int((clicked_points[0][1] + clicked_points[2][1]) / 2)
last_known_players = [
    [int((clicked_points[0][0]+clicked_points[1][0])/2) - 15, int((clicked_points[0][1]+mid_y)/2) - 25, 30, 50],
    [int((clicked_points[3][0]+clicked_points[2][0])/2) - 22, int((mid_y+clicked_points[3][1])/2) - 42, 45, 85]
]
player_velocities = [[0, 0], [0, 0]]
lost_frames_counter = [0, 0]

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_count % frame_skip != 0: continue
        
    frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
   
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=3, minSize=(20, 40), maxSize=(200, 250))
    
    valid_candidates = []
    for (x, y, w, h) in players:
        cx, cy = int(x + w / 2), int(y + h)
        inside_court = cv2.pointPolygonTest(court_polygon, (cx, cy), False)
        if inside_court >= 0:
            valid_candidates.append([x, y, w, h, cx, cy])

    tracked_this_frame = [None, None]

    if len(valid_candidates) > 0:
        dist_matrix = []
        for cand in valid_candidates:
            d_far = math.sqrt((cand[4] - (last_known_players[0][0] + last_known_players[0][2]/2))**2 + (cand[5] - (last_known_players[0][1] + last_known_players[0][3]))**2)
            d_near = math.sqrt((cand[4] - (last_known_players[1][0] + last_known_players[1][2]/2))**2 + (cand[5] - (last_known_players[1][1] + last_known_players[1][3]))**2)
            dist_matrix.append((cand, d_far, d_near))

        dist_matrix_far = sorted(dist_matrix, key=lambda item: item[1])
        dist_matrix_near = sorted(dist_matrix, key=lambda item: item[2])
        
        #red
        for item in dist_matrix_far:
            cand_box = item[0]
            if (cand_box[1] + cand_box[3]) <= mid_y : 
                tracked_this_frame[0] = cand_box[:4]
                break
                
       #white
        for item in dist_matrix_near:
            cand_box = item[0]
            if (cand_box[1] + cand_box[3]) > mid_y :
                if tracked_this_frame[0] is None or cand_box[:4] != tracked_this_frame[0]:
                    tracked_this_frame[1] = cand_box[:4]
                    break

    for idx in range(2):
        if tracked_this_frame[idx] is not None:
            lost_frames_counter[idx] = 0
            x_new, y_new, w_new, h_new = tracked_this_frame[idx]
            
            x_old, y_old, w_old, h_old = last_known_players[idx]
            
         
            x_calc = int(x_old * 0.55 + x_new * 0.45)
            y_calc = int(y_old * 0.55 + y_new * 0.45)
            w_calc = int(w_old * 0.55 + w_new * 0.45)
            h_calc = int(h_old * 0.55 + h_new * 0.45)
            
            player_velocities[idx] = [x_calc - x_old, y_calc - y_old]
            last_known_players[idx] = [x_calc, y_calc, w_calc, h_calc]
            
            cv2.rectangle(frame, (x_calc, y_calc), (x_calc + w_calc, y_calc + h_calc), (0, 255, 0), 2)
        else:
            lost_frames_counter[idx] += 1
            x_old, y_old, w_old, h_old = last_known_players[idx]
            dx, dy = player_velocities[idx]
            
            if lost_frames_counter[idx] < 30:
                dx = np.clip(dx, -5, 5)
                dy = np.clip(dy, -5, 5)
            else:
                dx, dy = int(dx * 0.4), int(dy * 0.4)
                
            x_calc = int(x_old + dx)
            y_calc = int(y_old + dy)
            w_calc, h_calc = w_old, h_old
            
            last_known_players[idx] = [x_calc, y_calc, w_calc, h_calc]
            player_velocities[idx] = [dx, dy]
            
            cv2.rectangle(frame, (x_calc, y_calc), (x_calc + w_calc, y_calc + h_calc), (0, 255, 0), 2)

        cx_foot = int(last_known_players[idx][0] + last_known_players[idx][2] / 2)
        cy_foot = int(last_known_players[idx][1] + last_known_players[idx][3])
        cv2.circle(frame, (cx_foot, cy_foot), 5, (0, 0, 255), -1)
        
        
        point_cam = np.array([[[cx_foot, cy_foot]]], dtype=np.float32)
        point_real = cv2.perspectiveTransform(point_cam, H)
        x_real, y_real = int(point_real[0][0][0]), int(point_real[0][0][1])
        
        if 0 <= x_real < SODO_W and 0 <= y_real < SODO_H:
            heatmap_matrix[y_real, x_real] += 15.0

    cv2.imshow("Giao dien Tracking ", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


court_bg = cv2.imread(court_bg_path)
if court_bg is not None:
    court_bg = cv2.resize(court_bg, (SODO_W, SODO_H))
else:
    court_bg = np.zeros((SODO_H, SODO_W, 3), dtype=np.uint8)
    court_bg[:] = (75, 95, 30)
    cv2.rectangle(court_bg, (15, 20), (SODO_W - 15, SODO_H - 20), (255, 255, 255), 2)
    cv2.line(court_bg, (15, SODO_H // 2), (SODO_W - 15, SODO_H // 2), (255, 255, 255), 2)

heatmap_blur = cv2.GaussianBlur(heatmap_matrix, (51, 51), 0)
heatmap_log = np.log1p(heatmap_blur)
heatmap_norm = cv2.normalize(heatmap_log, None, 0, 255, cv2.NORM_MINMAX)
heatmap_img = np.uint8(heatmap_norm)
heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)

court_mask = np.zeros((SODO_H, SODO_W), dtype=np.uint8)
cv2.rectangle(court_mask, (12, 15), (SODO_W - 12, SODO_H - 15), 255, -1)
heatmap_color_clipped = cv2.bitwise_and(heatmap_color, heatmap_color, mask=court_mask)

final_output = cv2.addWeighted(court_bg, 0.60, heatmap_color_clipped, 0.40, 0)
cv2.imwrite("heatmap_goc_nhin_tren_xuong.jpg", final_output)

cv2.imshow("Heatmap", final_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
