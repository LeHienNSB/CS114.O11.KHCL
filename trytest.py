import mediapipe as mp
import numpy as np
import pickle
import cv2
import copy
import itertools


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0 
    for index, landmark_point in enumerate(temp_landmark_list): #gán số thứ tự vào từng phần tử 
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # chuyển về thành mảng 1 chiều
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization các giá trị trong khoảng từ
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        if max_value != 0:
            return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


model_dict = pickle.load(open('./model1.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3,max_num_hands=2)
#các lable pridict thành chữ
labels_dict = {0: 'I/My', 1: 'You', 2: 'Where', 3: 'Store/Station', 4: 'Help', 5:'Eat', 6:'Drink', 7:'Home', 8:'Gas', 9: 'Need', 10:'Medicine', 
               11:'Police',12:'Direction', 13:'Transportation', 14:'Restroom', 15:'Call/Phone', 16:'Lost',17:'Keys', 18:'Right now', 19:'How'}


while True:

        
    #khai báo các list cần dùng
    data_aux = []
    x_ = []
    y_ = []
    aux=[]

    ret, frame = cap.read()

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    image_width, image_height = frame.shape[1], frame.shape[0]

    if results.multi_hand_landmarks:
    #2 vòng lặp cùng một lúc vừa vẽ landmark lên frame vừa detect frame
        for hand_landmarks in results.multi_hand_landmarks: #vòng lặp vẽ lên frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())   
        #thu thập data từ màn hình, không có lable    
        for hand_landmarks in results.multi_hand_landmarks: #vòng lặp để detect tay 
            if len(results.multi_hand_landmarks) == 2:
                for _, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    data_aux = np.array((landmark_x, landmark_y))
                    aux.append(data_aux)
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                temp_landmark_list = pre_process_landmark(aux)
            elif len(results.multi_hand_landmarks) == 1:
                for _, landmark in enumerate(hand_landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    data_aux = np.array((landmark_x, landmark_y))
                    aux.append(data_aux)
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                temp_landmark_list = pre_process_landmark(aux)
                for i in range (len(temp_landmark_list),84):
                    temp_landmark_list.append(0.0)
                for i in range(len(x_), 42):
                    x_.append(0)
                    y_.append(0)

        x = int(max(x_) * W) #giá trị để viết được chữ di chuyển theo tay chiều rộng (lấy max vì min có thể có giá trị 0=> không hiện lên màn hình)
        y = int(max(y_) * H) #giá trị để viết được chữ di chuyển theo tay chiều rộng

        prediction = model.predict([np.asarray(temp_landmark_list)])#cho data lấy được vào modle để predict

        predicted_character = labels_dict[int(prediction[0])]#từ data dịch về lable 

        cv2.putText(frame, predicted_character, (x - 200, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)#viết chữ trên frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)  == ord('q'):
            break
    else:
        cv2.putText(frame, 'Put your hand in the frame', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)#viết chữ trên frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)  == ord('q'):
                break
       
cap.release()#đóng cam
cv2.destroyAllWindows()#đóng cửa sổ