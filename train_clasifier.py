import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier # import thuật toán sử dụng từ module ensemble trong thư viện sklearn
from sklearn.model_selection import train_test_split #import class dùng để chia dữ liệu thành train và test
from sklearn.metrics import accuracy_score #import class dùng để đánh giá độ chính xác của máy


data_dict = pickle.load(open('./data.pickle', 'rb')) #giải tuần tự hóa module rồi load lên biến data_dict

#print (data_dict['labels'])

data = np.asarray(data_dict['data']) #chuyển list thành 1 array (mảng)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) #phân tách dữ liệu ra thành 2 phần random train và test với tỉ lệ 80-20
 
model = RandomForestClassifier() #sử dụng thuật toán randomforestclassifier làm model

model.fit(x_train, y_train) #fit giá trị

y_predict = model.predict(x_test) #giá trị dự đoán

score = accuracy_score(y_predict, y_test) # kiểm tra độ chính xác của giá trị dự đoán với giá trị thực 

print('{}% of samples were classified correctly !'.format(score * 100)) #print ra tỉ lệ dự đoán đúng sai


f = open('model.p', 'wb') 
pickle.dump({'model': model}, f) #tuần tự hóa lại module rồi lưu vapf file model.p
f.close() #đóng file
