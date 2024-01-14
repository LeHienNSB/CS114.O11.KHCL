import os 
import pickle #module dùng để tuần tự hóa thành 1 luồng byte và giải tuần tự hóa
import itertools
import mediapipe as mp #bộ khung gồm những giải pháp được xây dựng sẵn 
import cv2 
import copy #thư viện để copy file
import numpy as np


mp_hands = mp.solutions.hands #khởi tạo lớp hands và lưu vào biến mp_hands
mp_drawing = mp.solutions.drawing_utils  # thiết lập hàm vẽ điểm mốc trên ảnh tay 
mp_drawing_styles = mp.solutions.drawing_styles 

hands = mp_hands.Hands(static_image_mode=True , min_detection_confidence=0.3) #thiết lập chức năng giữ các điểm mốc của tay

DATA_DIR='./data_200'

data = [] #tạo list data để lưu ảnh của mỗi tệp
labels = [] #tạo list lable để lưu nhãn

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list) #tạo một list giống list cũ nhưng ở địa chỉ khác hoàn toàn 

    # chuyển các tọa độ của từng mốc thành khoảng cách các mốc so với điểm gốc
    base_x, base_y = 0, 0 
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0: #nếu là điểm gốc
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x  #Lấy phần tử thứ 1 của mảng index
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y  #Lấy phần tử thứ 2 của mạng index

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization để các phần tử giá trị chỉ từ -1 đến 1 
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        if max_value !=0:
            return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


for dir_ in os.listdir(DATA_DIR): #chạy từng tệp trong thư mục data (0,1,2...)
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): #chạy từng ảnh một trong 1 tệp
        data_aux = [] #tạo list để lưu data của mỗi ảnh trong tệp
        aux=[]
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path)) #đọc ảnh trong thư mục 
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #chuyển từ BGR sang RGB (format để làm input cho mediapipe)
        image_width, image_height = img.shape[1], img.shape[0]
        results = hands.process(img_rgb) #kết quả là landmark được đọc ra trong ảnh
        if len(results.multi_hand_landmarks) == 2 and results.multi_hand_landmarks: #check xem kết quả có được phát hiện hay không
            for hand_landmarks in results.multi_hand_landmarks: #vòng lặp số bàn tay
                for _, landmark in enumerate(hand_landmarks.landmark): #vòng lặp từng điểm trên bàn tay
                    landmark_x = min(int(landmark.x * image_width), image_width - 1) #đề phòng nếu có ảnh mà tay nằm ngoài ảnh về chiều rộng
                    landmark_y = min(int(landmark.y * image_height), image_height - 1) #đề phòng nếu có ảnh mà tay nằm ngoài ảnh về chiều dài
                    data_aux = np.array((landmark_x, landmark_y)) #chuyển x và y thành dạng mảng
                    aux.append(data_aux) #gộp các tọa độ lại theo nhóm mốc
                temp_landmark_list = pre_process_landmark(aux) #list sau khi được chuyển giá trị và normalize 
            data.append(temp_landmark_list)
            labels.append(dir_)
        elif len(results.multi_hand_landmarks) == 1 and results.multi_hand_landmarks: #check xem kết quả có được phát hiện hay không
            for hand_landmarks in results.multi_hand_landmarks: #vòng lặp số bàn tay
                for _, landmark in enumerate(hand_landmarks.landmark):#vòng lặp từng điểm trên bàn tay
                    landmark_x = min(int(landmark.x * image_width), image_width - 1) #đề phòng nếu có ảnh mà tay nằm ngoài ảnh về chiều rộng
                    landmark_y = min(int(landmark.y * image_height), image_height - 1) #đề phòng nếu có ảnh mà tay nằm ngoài ảnh về chiều dài
                    data_aux = np.array((landmark_x, landmark_y))#chuyển x và y thành dạng mảng
                    aux.append(data_aux)
                temp_landmark_list = pre_process_landmark(aux)
                for i in range (len(temp_landmark_list),84): #nếu bộ dữ liệu chưa đủ 84 phần tử thì thêm các phần tử còn thiếu là 0
                    temp_landmark_list.append(0.0)
            data.append(temp_landmark_list)
            labels.append(dir_)

f = open('data.pickle', 'wb') 
pickle.dump({'data': data, 'labels': labels}, f) #tuần tự hóa module và ghi vào tệp data.pickle
f.close() #đóng file  