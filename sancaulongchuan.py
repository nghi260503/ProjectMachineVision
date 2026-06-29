import cv2
import numpy as np

# Định nghĩa kích thước Sân Dọc: Rộng (Width) = 410px, Cao (Height) = 640px
WIDTH = 410
HEIGHT = 640

# 1. ĐÃ ĐỔI: Tạo khung ảnh nền màu xanh thảm sân sáng, tươi và chuyên nghiệp (Định dạng BGR)
court = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
court[:] = (65, 80, 20)  # Mã màu xanh lá cây sáng, tươi chuẩn thảm Yonex

WHITE = (255, 255, 255)
THICKNESS = 2

# Lề biên (Padding) từ mép thảm vào đến đường biên ngoài cùng
pad_x = 20
pad_y = 30

# Tính toán các mốc tọa độ vùng thi đấu thực tế theo tỷ lệ chuẩn
vach_ngoai_w = WIDTH - 2 * pad_x   # Chiều rộng vùng thi đấu (tương đương 6.1m thực tế)
vach_ngoai_h = HEIGHT - 2 * pad_y  # Chiều dài vùng thi đấu (tương đương 13.4m thực tế)

x_left = pad_x
x_right = WIDTH - pad_x
y_top = pad_y
y_bottom = HEIGHT - pad_y

# 2. TIẾN HÀNH VẼ CÁC ĐƯỜNG KẺ SÂN DỌC TIÊU CHUẨN
# Vẽ đường biên tổng chữ nhật ngoài cùng
cv2.rectangle(court, (x_left, y_top), (x_right, y_bottom), WHITE, THICKNESS)

# Vẽ đường lưới chính giữa sân (chia đôi chiều cao/chiều dài sân)
y_center = HEIGHT // 2
cv2.line(court, (x_left, y_center), (x_right, y_center), WHITE, THICKNESS)

# Vẽ vạch giao cầu ngắn (Cách lưới đúng 1.98m thực tế)
dist_short_service = int(1.98 / 13.4 * vach_ngoai_h)
cv2.line(court, (x_left, y_center - dist_short_service), (x_right, y_center - dist_short_service), WHITE, THICKNESS)
cv2.line(court, (x_left, y_center + dist_short_service), (x_right, y_center + dist_short_service), WHITE, THICKNESS)

# Vẽ vạch giao cầu dài cho góc đánh đôi (Cách biên dọc cuối sân đúng 0.76m thực tế)
dist_long_service = int(0.76 / 13.4 * vach_ngoai_h)
cv2.line(court, (x_left, y_top + dist_long_service), (x_right, y_top + dist_long_service), WHITE, THICKNESS)
cv2.line(court, (x_left, y_bottom - dist_long_service), (x_right, y_bottom - dist_long_service), WHITE, THICKNESS)

# Vẽ đường biên dọc trong cho đánh đơn (Cách biên ngoài hai bên đúng 0.46m thực tế)
dist_sidebar = int(0.46 / 6.1 * vach_ngoai_w)
cv2.line(court, (x_left + dist_sidebar, y_top), (x_left + dist_sidebar, y_bottom), WHITE, THICKNESS)
cv2.line(court, (x_right - dist_sidebar, y_top), (x_right - dist_sidebar, y_bottom), WHITE, THICKNESS)

# Vẽ đường trung tâm chia đôi ô giao cầu trái/phải (Vẽ dọc từ vạch ngắn đến biên cuối)
x_center = WIDTH // 2
cv2.line(court, (x_center, y_top), (x_center, y_center - dist_short_service), WHITE, THICKNESS)
cv2.line(court, (x_center, y_center + dist_short_service), (x_center, y_bottom), WHITE, THICKNESS)

# 3. LƯU VÀ HIỂN THỊ SƠ ĐỒ DỌC
cv2.imwrite('san_cau_long_doc.jpg', court)
cv2.imshow('So do san cau long doc', court)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Đã tạo thành công ảnh sân dọc sáng 'san_cau_long_doc.jpg' chuẩn tỷ lệ để chạy mô hình!")