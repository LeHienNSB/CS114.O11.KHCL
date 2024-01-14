import os
import cv2


DATA_DIR = './data1'

number_of_classes = 1
dataset_size = 200

cap = cv2.VideoCapture(0)
name = 1


if not os.path.exists(os.path.join(DATA_DIR, str(name))):
    os.makedirs(os.path.join(DATA_DIR, str(name)))

print('Collecting data for class {}'.format(name))

done = False
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        cv2.waitKey(3000)
        break

counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(50)
    cv2.imwrite(os.path.join(DATA_DIR, str(name), '{}.jpg'.format(counter)), frame)

    counter += 1

cap.release()
cv2.destroyAllWindows()