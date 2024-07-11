from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#Thiết lập ngưỡng độ tin cậy cho việc hiển thị các hộp giới hạn.
confident = 0.7

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video


model = YOLO("../models/n_version_4_5.pt")

classNames = ["real", "fake"]
#Khởi tạo hai biến để tính toán FPS.
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    #Đọc một khung hình từ webcam. success là boolean cho biết liệu khung hình có được lấy thành công hay không, img là khung hình đã chụp
    success, img = cap.read()
    #Sử dụng mô hình YOLO để phát hiện các đối tượng trong khung hình.
    results = model(img, stream=True)
    for r in results:
        #Lấy danh sách các hộp giới hạn (bounding boxes) từ kết quả phát hiện.
        boxes = r.boxes
        #Lặp qua từng hộp giới hạn
        for box in boxes:
            #Lấy tọa độ của hộp giới hạn.
            x1, y1, x2, y2 = box.xyxy[0]
            #Chuyển đổi tọa độ sang kiểu số nguyên.
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #Tính toán chiều rộng và chiều cao của hộp giới hạn.
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            #Kiểm tra xem độ tin cậy có lớn hơn ngưỡng confident hay không.
            if conf > confident:
                if classNames[cls] == 'real':
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                # Vẽ hộp giới hạn với màu đã chọn.
                cvzone.cornerRect(img, (x1, y1, w, h),colorC= color,colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB= color,colorR=color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("FPS:", fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)