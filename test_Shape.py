import os 
import mediapipe as mp #bộ khung gồm những giải pháp được xây dựng sẵn 
import cv2 
import numpy as np


mp_hands = mp.solutions.hands #khởi tạo lớp hands và lưu vào biến mp_hands
mp_drawing = mp.solutions.drawing_utils  # thiết lập hàm vẽ điểm mốc trên ảnh tay 
mp_drawing_styles = mp.solutions.drawing_styles 

hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3) #thiết lập chức năng giữ các điểm mốc của tay

DATA_DIR='./data/1'

data = [] #tạo list data để lưu ảnh của mỗi tệp

data_aux = [] #tạo list để lưu data của mỗi ảnh trong tệp
for img_path in os.listdir(os.path.join(DATA_DIR)):
        data_aux = [] #tạo list để lưu data của mỗi ảnh trong tệp
        img = cv2.imread(os.path.join(DATA_DIR, img_path)) #đọc ảnh trong thư mục 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #chuyển từ BGR sang RGB (format để làm input cho mediapipe)
        
        results = hands.process(img_rgb) #kết quả là landmark được đọc ra trong ảnh
        if results.multi_hand_landmarks: #check xem kết quả có được phát hiện hay không
            for hand_landmarks in results.multi_hand_landmarks: #vòng lặp qua tất cả các điểm được phát hiện trong ảnh
                    for i in range(len(hand_landmarks.landmark)): # các mốc điểm trong mỗi mảng 
                        x = hand_landmarks.landmark[i].x #độ rộng
                        y = hand_landmarks.landmark[i].y #độ cao 
                        data_aux.append(x)
                        data_aux.append(y)


            data.append(data_aux)


print (np.array(data).shape)