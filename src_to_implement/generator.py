import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.transform
import math


# with open('Labels.json','r') as file:
#     str1 = file.read()
#     data = json.loads(str1)
#     print(data)

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:

    def __init__(self, file_path , label_path, batch_size, image_size, rotation=bool, mirroring=bool, shuffle=bool):

        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        # We made a counter.
        # count计数器 add 用来记录总共执行了多少次next，//The count counter add used to record how many times in total performed next.
        # 当执行epoch更新之后，batch_count又要归零。// When do epoch after the update, batch_count again to return to zero.
        self.batch_count=0
        self.epoch = 0

        # filenames里面存储的是exercise_data文件夹下面的文件名，类型是列表，里面的数据是字符串//
        # Filenames storage is exercise_data folder inside the following file name, type is the list, the inside of the data as a string
        self.filenames = os.listdir(file_path)

        # num_files存储.npy文件总数量
        # The total number num_files storage. Npy file
        self.num_files=len(self.filenames)




        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # 存储next函数每次调用返回的文件，数据格式为(batch_size,48,48,3)
        # Stored next to each invocation returns the file, the data format for (batch_size, 48,48,3)
        self.images = None
        # 存储返回的images图片所对应的类别，数据格式为(batch_size,)
        #Store the returned images images corresponding to category, data format for (batch_size,)
        self.labels = []
        # 用于存储从json文件中解析出来的文件名和类对应关系的字典
        # Used to store the parsed from the json file of file names and corresponding relation of the dictionary
        self.dic=None


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method

        # add
        # self.batch_count+=1

        if self.shuffle is True:
            np.random.shuffle(self.filenames)

        self.batch_number = self.num_files // self.batch_size
        c = self.num_files % self.batch_size
        # 计算c是为了构造一个完整的能够被batch_size整除的数据集，如果刚开始batch_size能够整除num_files，那么就可以使用原来的数据集
        # C is calculated in order to construct a complete data set can be divided exactly by batch_size
        # if initially batch_size divisible num_files, so you can use the original data set
        # 如果不能整除，则从数据集开头再添加余数个数据，并构成新的数据集dict_and
# If it cannot be divisible from the data set is beginning to add remainder data, and constitute a new dict_and data sets
        if c > 0:
            self.batch_number += 1
            suplement=self.filenames[: self.batch_size - c]
            self.filenames.extend(suplement)

        self.new_dataset_len=len(self.filenames)

        central=range(self.new_dataset_len)

        # 为新数据集构造一个字典
        # Constructing a dictionary for the new data set
        self.dict_and=dict(zip(central,self.filenames))



        with open(self.label_path, 'r') as file:
            temp = file.read()
            self.dic = json.loads(temp)

        images=[]
        # count()
        # 作出切片 读取count 在当前batch 和batch +1中截取
        # Make a slice read count interception in the current batch and batch + 1
        times=self.batch_count%self.batch_number
        # 如果经过了一次整除，那么更新epoch
        # if can be divided exactly by directly, then update the epoch
        # if self.batch_count>0 and times==0:
        #     self.epoch+=1
        self.epoch=(self.batch_count*self.batch_size)//self.num_files
        # for j in range(self.batch_number):

# 切分出当前的batch 并且遍历
        # Split out the current batch and traversal
        for i in range(self.batch_size*times,self.batch_size*(times+1)):
            image = np.load(self.file_path + str(self.dict_and[i]))
            # image = skimage.transform.resize(image, self.image_size, mode="constant")
            images.append(image)
#将当前batch的每张图片 都放在列表里面
# The current batch of each image in the list
            self.labels.append(self.dic[self.dict_and[i][:-4]])

        # self.images=np.array(images)
        # 建立
        # resized=np.asarray(list())
        # self.images=np.asarray(self.images)
## 用for循环 通过遍历长度 对当前batch每张图片进行大小的调整。 调整图片
# Use a for loop by traversing the length for the current batch to adjust the size of each image.Adjust the picture

        self.images=[]
        for i in range(len(images)):
            resized_img=skimage.transform.resize(np.asarray(images[i]),self.image_size,mode="constant")
            self.images.append(resized_img)

        # images = np.array(images[(self.batch_count-1)*self.batch_size:self.batch_count*self.batch_size])
## 通过调用augment 函数 对图片进行随机翻转
        # Random images flip through a call to augment function
        flip_rotate_img=[]
        self.images=np.asarray(self.images)
        for img in self.images:
            flip_rotate_img.append(self.augment(img))
        #
        self.images=flip_rotate_img

        self.batch_count += 1
# 计数器记录当前batch 数量，我们当是第几次batch
        # Counter to record the batch number,to know we now is which batch
        # return np.array(images), self.labels
        return np.asarray(self.images), self.labels


    # haungjin
    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        # self_ImageGenerator.next()
        img_1=img
## 设置一个随机数（0。1）， 当mirrtor是true 的时候，并且当它小于1/3 我们执行上下翻转
# Set a random number (0.1), when mirrtor is true, and when it is less than a third we perform upside down
# Flip array in the left/right direction.
# flip
# Flip array in one or more dimensions.
        if self.mirroring is True :
            if random.random() < (1/3) :
               img = np.fliplr(img)
               # label[:, 1] = 1 - label[:, 1]
            elif (1/3) < random.random() < (2/3) :
               img = np.flipud(img)
            else:
                img=img
## same way rotation 不同随机数 翻转角度
        if self.rotation is True:
            if random.random() < 1/4 :
               img = np.rot90(img,1)
               # label[:, 1] = 1 - label[:, 1]
            elif 1/4 < random.random() < 2/4 :
               img = np.rot90(img,2)
               # label[:, 2] = 1 - label[:, 2]
            elif 2/4 < random.random() < 3/4 :
                img = np.rot90(img,3)
               # label[:, 2] = 1 - label[:, 2]
            else :
                img = img

        return img
## we made it in next
    def current_epoch(self):
        # print()
        # # if self.batch_number * self.batch_size + self.batch_size > len(os.listdir(self.file_path)):
        # #     return self.epoch + 1
        # # else:
        # #     return self.epoch
        # in
        # if self.batch_count*self.batch_size>len(os.listdir(self.file_path)):
        #     self.batch_count=0
        #     self.epoch+=1
        return self.epoch


    def class_name(self, x):#output is Str
        # This function returns the class name for a specific input
        # TODO: implement class name function
        return self.class_dict[x]



    # def show(self):
    #     #In order to verify that the generator creates batches as required, this functions calls next to get a
    #     # batch of images and labels and visualizes it.
    #     #TODO: implement show method
    #     self.next()
    #     batch=[]
    #
    #     for i in range(self.batch_size):
    #         plt.subplot(1, self.batch_size, 1)
    #         plt.axis("off")
    #         plt.title(self.class_name(self.labels))
    #         plt.imshow(self.images[i])
    #         batch.append(self.images)
    #     plt.subplot(1, self.batch_size, 1)
    #     plt.imshow(batch)

#
# if __name__=="__main__":
# # def main():
#     file_path="./exercise_data/"
#     label_path="./Labels.json"
#     batch_size=16
#     image_size=(48,48,3)
#
#     gen1=ImageGenerator(file_path,label_path,batch_size,image_size,shuffle=True,rotation=True,mirroring=True)
#     one=gen1.next()
#     print()



    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        #因为是测试程序，所以并没有打开镜像和旋转开关
        # newGenObject=ImageGenerator(self.file_path,self.label_path,self.batch_size,self.image_size,shuffle=self.shuffle,rotation=self.rotation,)
        imgLabTuple=self.next()

        #检查输出的数据图片格式是否符合要求
        # Check the output data of image format is in accordance with the requirements

        # print("desired_image_size:" + str(self.image_size))
        # print("dataset_image_size:" + str(np.asarray(self.npyDatabank[0]).shape))
        # print("next()_image_size:" + str(imgLabTuple[0][0].shape))
        fig=plt.figure()
        # 控制一行显示多少张图
        # Control the row shows how many picture
        colmNum=3
        rowNum=math.ceil(self.batch_size/colmNum)
        for i in range(len(imgLabTuple[0])):
            fig.add_subplot(rowNum,colmNum,i+1)
            plt.imshow(imgLabTuple[0][i])
            plt.title(self.class_name(imgLabTuple[1][i]))
            plt.axis("off")
        plt.show()



if __name__=="__main__":
# def main():
#
    file_path="./exercise_data/"
    label_path="./Labels.json"
    batch_size=12
    image_size=(48,48,3)

    gen1=ImageGenerator(file_path,label_path,batch_size,image_size,shuffle=True,rotation=True,mirroring=True)
#迭代次数 Iterations
    iterTimes=1
    count=0
    # ImageGenerator.show(gen1)
    gen1.show()






