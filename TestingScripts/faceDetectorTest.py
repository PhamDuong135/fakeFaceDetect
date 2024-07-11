#Test cho trình phát hiện khuôn mặt
#Sử dụng trình phát hiện gương mặt được cung cấp bởi Google
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

# Initialize the webcam
# '2' means the third camera connected to the computer, usually 0 refers to the built-in webcam
cap = cv2.VideoCapture(0)

# Initialize the FaceDetector object
# minDetectionCon: Ngưỡng tin cậy phát hiện tối thiểu
detector = FaceDetector(minDetectionCon=0.5)

# Chạy vòng lặp để liên tục nhận khung hình từ webcam
while True:
    # Đọc khung hình hiện tại từ webcam
    # success: Boolean, liệu khung có được lấy thành công hay không
    # img: khung hình đã chụp
    success, img = cap.read()
        # Phát hiện khuôn mặt trong ảnh
        # img: Hình ảnh được cập nhật
        # bboxs: Danh sách các hộp giới hạn xung quanh các khuôn mặt được phát hiện
    img, bboxs = detector.findFaces(img, draw=False)
        # Kiểm tra xem có phát hiện được khuôn mặt nào không
    if bboxs:
        # Lặp qua từng hộp giới hạn
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'

            # ---- Lấy dữ liệu  ---- #
            center = bbox["center"]
            #Lấy tọa độ và kích thước của hộp giới hạn (tọa độ góc trên bên trái (x, y) và chiều rộng (w), chiều cao (h)).
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # ---- Draw Data  ---- #
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10))
            cvzone.cornerRect(img, (x, y, w, h))

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)

