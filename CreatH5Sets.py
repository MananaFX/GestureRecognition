import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
def get_files(file_dir, label_Gesture9=None):
    Gesture0 = []
    label_Gesture0 = []
    Gesture1 = []
    label_Gesture1 = []
    Gesture2 = []
    label_Gesture2 = []
    Gesture3 = []
    label_Gesture3 = []
    Gesture4 = []
    label_Gesture4 = []
    Gesture5 = []
    label_Gesture5 = []
    Gesture6 = []
    label_Gesture6 = []
    Gesture7 = []
    label_Gesture7 = []
    Gesture8 = []
    label_Gesture8 = []
    Gesture9 = []
    label_Gesture9 = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='Gestures0':
            Gesture0.append(file_dir+file)
            label_Gesture0.append(0)
        elif name[0] =='Gesture1':
            Gesture1.append(file_dir+file)
            label_Gesture1.append(1)
        elif name[0]=='Gesture2':
            Gesture2.append(file_dir+file)
            label_Gesture2.append(2)
        elif name[0]=='Gesture3':
            Gesture3.append(file_dir+file)
            label_Gesture3.append(3)
        elif name[0] == 'Gesture4':
            Gesture4.append(file_dir + file)
            label_Gesture4.append(4)
        elif name[0] == 'Gesture5':
            Gesture5.append(file_dir + file)
            label_Gesture5.append(5)
        elif  name[0] == 'Gesture6':
            Gesture6.append(file_dir + file)
            label_Gesture6.append(6)
        elif  name[0] == 'Gesture7':
            Gesture7.append(file_dir + file)
            label_Gesture7.append(7)
        elif  name[0] == 'Gesture8':
            Gesture8.append(file_dir + file)
            label_Gesture8.append(8)
        else:
            Gesture9.append(file_dir + file)
            label_Gesture9.append(9)



    image_list = np.hstack((Gesture0,Gesture1,Gesture2,Gesture3,Gesture4,Gesture5,Gesture6,Gesture7,Gesture8,Gesture9))
    label_list = np.hstack((label_Gesture0,label_Gesture1,label_Gesture2,label_Gesture3,label_Gesture4,label_Gesture5,label_Gesture6,label_Gesture7,label_Gesture8,label_Gesture9))


    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    print("图像标签化完成")
    return  image_list,label_list

train_dir = 'Gesture/'
image_list,label_list = get_files(train_dir)

print('The length of train images'+str(len(image_list)))
print('The length of train labels'+str(len(label_list)))


m=len(image_list)
m_train=int(0.8*m)
m_test=m-m_train

Train_image =  np.random.rand(m_train, 1920, 1440, 3).astype('float32')
Train_label = np.random.rand(m_train, 1).astype('float32')

Test_image =  np.random.rand(m_test, 1920, 1440, 3).astype('float32')
Test_label = np.random.rand(m_test, 1).astype('float32')
for i in range(m_train):
    Train_image[i] = np.array(plt.imread(image_list[i]))
    Train_label[i] = np.array(label_list[i])

f = h5py.File('datasets\Owndata.h5', 'w')
f.create_dataset('X_test', data=Train_image)
f.create_dataset('y_test', data=Train_label)
f.close()


