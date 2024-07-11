import os
import random
import shutil
from itertools import islice


outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/All"
splitRatio = {"train":0.7,"val":0.2,"test":0.1}
classes = ["real","fake"]

try:
    #xóa thư mục và mọi thứ trong đó để lần khởi chạy sau dữ liệu ở trong không bị lặp lại
    shutil.rmtree(outputFolderPath)
    print("Remove Directory")
except OSError as e:
    os.mkdir(outputFolderPath)

# Tạo các thư mục con để chứa ảnh và nhãn cho từng tập dữ liệu.
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)

#Lấy danh sách các tên tệp
listNames = os.listdir(inputFolderPath)
uniqueNames = []
#Tạo một danh sách chứa các tên tệp duy nhất bằng cách loại bỏ phần mở rộng (ví dụ .jpg hoặc .txt).
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))

# Xáo trộn ngẫu nhiên danh sách các tên tệp để chia dữ liệu.
random.shuffle(uniqueNames)


#Tính toán số lượng ảnh cho từng tập
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])


#Phân bổ ảnh dư cho tập huấn luyện nếu cần thiết
#Nếu tổng số lượng ảnh không chia hết cho tổng số lượng các tập (train + val + test),
# thì số lượng ảnh của tập train sẽ được cộng thêm để đảm bảo không bỏ sót ảnh nào.
if lenData != lenTrain+lenTest+lenVal:
    remaining = lenData-(lenTrain+lenTest+lenVal)
    lenTrain += remaining

#Chia danh sách tên tệp thành các tập dữ liệu
lengthToSplit = [lenTrain,lenVal,lenTest] #Danh sách số lượng ảnh cần chia cho mỗi tập (train, val, test).
# Sử dụng islice để chia danh sách uniqueNames thành các tập con dựa trên lengthToSplit.
Input = iter(uniqueNames)
Output = [list(islice(Input,elem))for elem in lengthToSplit]
print(f'Total Images: {lenData} \n Spit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')



#Sao chép các tệp ảnh và nhãn vào các thư mục tương ứng
sequence = ['train','val','test'] #Danh sách tên các tập (train, val, test).
#Dùng vòng lặp để sao chép từng tệp ảnh và nhãn từ thư mục inputFolderPath sang thư mục tương ứng trong outputFolderPath.
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")

#----------Tao file DataYaml--------------
#Chuỗi YAML để chỉ định đường dẫn và các thông số cho mô hình YOLO.
#Ghi chuỗi này vào tệp data.yaml trong thư mục outputFolderPath.
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names : {classes}'


f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()




