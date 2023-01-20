import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from matplotlib import cm as mplcm


class Checker:
    def __init__(self,resolution,tile_size):
        ##instantiate
        self.resolution=resolution
        self.tile_size=tile_size
        self.output=None

    # @staticmethod
    def draw(self):
## 生成全1矩阵  在 imshow黑色画布
        c = np.ones((2*self.tile_size,2*self.tile_size),dtype='int')
## 扩C
        t = self.resolution // (2 * self.tile_size)
        # t = int(t)
# 纵轴从头开始取tile size 长度 右上角变成黑色 左下 通过步长
        c[:self.tile_size,:self.tile_size]= 0
        c[self.tile_size:,self.tile_size:] = 0
        # print(c)
## 变成t倍
        self.output = np.tile(c,(t,t))
        # print(c)
        return self.output.copy()


    #tile_size = 25
    #esolution = 250
    #print(output)



    def show(self):
        plt.imshow(self.output,cmap='gray',interpolation='nearest')
        plt.show()




class Circle():
    def __init__(self, resolution, radius, position):
        # We defined the three parameters.In order to define the image of the width of each how many pixels
        # number of pixels
        self.resolution = resolution
        self.radius = radius
        # position is a tuple
        ## Center position
        self.position = position
        self.output = None
## 0
    def draw(self):
        x = np.arange(0,self.resolution)
        y = np.arange(0,self.resolution)
## 先生成两个数组
        #meshgrid because we want to tranverse all dots in a 2-D coordinates
## 通过x,y 拓展成一个白画布
        X, Y = np.meshgrid(x, y)
        posOfCenter = [self.position[0], self.position[1]]
        ## limit conditions
        # ##圆的范围
        locations = (X - posOfCenter[0]) ** 2 + (Y - posOfCenter[1]) ** 2 <= self.radius ** 2
        self.output = np.zeros(X.shape,dtype=bool)
        self.output[locations] = True

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.show()




class Spectrum:
## We use the method of joining together.
    def __init__(self,resolution):
        self.resolution=resolution
        self.output=None
    def draw(self):
        spectrum = np.ones([self.resolution, self.resolution , 3], dtype=np.uint8)  # init the array
        ## 0 255 gradual change
        #green
    ## //等差数列 0255控制渐变颜色
        spectrum[:, :, 1] = np.linspace(0,255, self.resolution)
        spectrum = spectrum.swapaxes(0, 1)
        ## blue
        spectrum[:, :, 2] = np.linspace(255, 0, self.resolution)
        ## red
        spectrum[:, :, 0] = np.linspace(0, 255, self.resolution)
        self.output=spectrum
        return self.output.copy()/255
    def show(self):
        plt.imshow(self.output)
        plt.xticks([])
        plt.yticks([])
        plt.show()
















