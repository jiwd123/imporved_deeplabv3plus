from pickle import NONE
from imutils import paths
from yaml import scan
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image, ImageEnhance
import sys,os
from deeplab import DeeplabV3
from deeplab1 import DeeplabV31
import datetime
import time
from nets.makedir import mkdir
from nets.deldir import deldir
from nets.tpfg import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import nets.twain_module as twain_module

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images",default="E:/github/deeplabv3-plus-pytorch-main/img/", type=str, #required=True,
                help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", default="E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/out3/",type=str, #required=True,
                help="path to the output image")
ap.add_argument("-c", "--cache", default="E:/github/deeplabv3-plus-pytorch-main/VOCdevkit/VOC2007/",type=str, #required=True,
                help="path to the cache")
ap.add_argument("-n", "--name", default=b'scan1',type=str, #required=True, 默认None
                help="name of scan")
ap.add_argument("-m", "--num", default=3,type=str, #required=True,
                help="num of scan")
ap.add_argument("-t", "--time", default=1/60,type=str, #required=True,
                help="time interval")
args = vars(ap.parse_args())  # vars函数是实现返回对象object的属性和属性值的字典对象

print(args)  # {python cdpj.py --images E:/datasetroot/img/ --output E:/datasetroot/outimg/ --cache E:/datasetroot/}
# 匹配输入图像的路径并初始化我们的图像列表
# rectangular_region = 2
Image.MAX_IMAGE_PIXELS = 2300000000




print("[INFO] loading images...")
if __name__ == "__main__":
    mkdir(args['images'])
    mkdir(args['output'])
    mkdir(args['cache'] + '/out/')
    mkdir(args['cache'] + '/out1/')
    mkdir(args['cache'] + 'out2/')
    mkdir(args['cache'] + '/imgout/')
    mkdir(args['cache'] + '/imgout1/')
    deeplab = DeeplabV3()
    deeplab1 = DeeplabV31()
    count = False
    name_classes = ["background","root"]
    if args['name'] == None:
        print("无可用扫描仪")
        pass
    else:
        nsc = 1
        for ic in range(0,args['num']):
            twain_module.acquire(args['images'] + str(nsc) + '.jpg', ds_name=args['name'], dpi=300,pixel_type="color") # 设置dpi300,彩色模式
            t1 = datetime.datetime.now().microsecond
            t2 = time.mktime(datetime.datetime.now().timetuple())
            image = Image.open(args['images'] + str(nsc) + '.jpg')
            # 图像整体预测
            f_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
            f_image.save(args['cache'] + '/imgout/' + str(nsc) + '.jpg')
            f_image = cv2.imread(args['cache'] + '/imgout/' + str(nsc) + '.jpg')
            image_name = str(nsc)
            clip(f_image, 512, image_name, args['cache'] + '/imgout1/')

            # 图像分割成预测
            
            image = cv2.imread(args['images'] + str(nsc) + '.jpg')
            
            clip(image, 512, image_name, args['cache'] + '/out/')
            f = os.listdir(args['cache'] + '/out/')
            n = 0
            i = 0
            for i in f:
                a = Image.open(args['cache'] + '/out/' + f[n])   
                b = cv2.imread(args['cache'] + '/imgout1/' + f[n])  
                if np.all(b == 0):
                    r_image = Image.new('RGB',(512,512),(0,0,0)) 
                    
                else:
                    
                    r_image = deeplab1.detect_image(a, count=count, name_classes=name_classes)

                #r_image.show()
                r_image.save(args['cache'] + '/out1/' + f[n])
                n = n + 1
            IMAGES_PATH = args['cache'] + '/out1/' # 图片集地址
            IMAGES_FORMAT = ['.jpg', '.jpg'] # 图片格式
            height, weight = image.shape[:2]

            IMAGE_SIZE = 512
            IMAGE_ROW = 20 # 图片间隔，也就是合并成一张图后，一共有几行
            IMAGE_COLUMN = 28 # 图片间隔，也就是合并成一张图后，一共有几列
            IMAGE_SAVE_PATH = args['cache'] + '/out2/' + str(nsc) + '.jpg' # 图片转换后的地址
            to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
            indey = 0

            for ii in range(0,IMAGE_ROW*IMAGE_COLUMN):
                if indey < 20:
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, (indey*IMAGE_SIZE, 0*IMAGE_SIZE))
                elif (indey >=20) & (indey < 40):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-20)*IMAGE_SIZE, 1*IMAGE_SIZE))
                elif (indey >=40) & (indey < 60):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-40)*IMAGE_SIZE, 2*IMAGE_SIZE))
                elif (indey >=60) & (indey < 80):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-60)*IMAGE_SIZE, 3*IMAGE_SIZE))
                elif (indey >=80) & (indey < 100):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-80)*IMAGE_SIZE, 4*IMAGE_SIZE))
                elif (indey >=100) & (indey < 120):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-100)*IMAGE_SIZE, 5*IMAGE_SIZE))
                elif (indey >=120) & (indey < 140):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-120)*IMAGE_SIZE, 6*IMAGE_SIZE))
                elif (indey >=140) & (indey < 160):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-140)*IMAGE_SIZE, 7*IMAGE_SIZE))
                elif (indey >=160) & (indey < 180):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-160)*IMAGE_SIZE, 8*IMAGE_SIZE))
                elif (indey >=180) & (indey < 200):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-180)*IMAGE_SIZE, 9*IMAGE_SIZE))
                elif (indey >=200) & (indey < 220):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-200)*IMAGE_SIZE, 10*IMAGE_SIZE))
                elif (indey >=220) & (indey < 240):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-220)*IMAGE_SIZE, 11*IMAGE_SIZE))
                elif (indey >=240) & (indey < 260):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-240)*IMAGE_SIZE, 12*IMAGE_SIZE))
                elif (indey >=260) & (indey < 280):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-260)*IMAGE_SIZE, 13*IMAGE_SIZE))
                elif (indey >=280) & (indey < 300):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-280)*IMAGE_SIZE, 14*IMAGE_SIZE))
                elif (indey >=300) & (indey < 320):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-300)*IMAGE_SIZE, 15*IMAGE_SIZE))
                elif (indey >=320) & (indey < 340):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-320)*IMAGE_SIZE, 16*IMAGE_SIZE))
                elif (indey >=340) & (indey < 360):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-340)*IMAGE_SIZE, 17*IMAGE_SIZE))
                elif (indey >=360) & (indey < 380):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-360)*IMAGE_SIZE, 18*IMAGE_SIZE))
                elif (indey >=380) & (indey < 400):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-380)*IMAGE_SIZE, 19*IMAGE_SIZE))
                elif (indey >=400) & (indey < 420):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-400)*IMAGE_SIZE, 20*IMAGE_SIZE))
                elif (indey >=420) & (indey < 440):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-420)*IMAGE_SIZE, 21*IMAGE_SIZE))
                elif (indey >=440) & (indey < 460):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-440)*IMAGE_SIZE, 22*IMAGE_SIZE))
                elif (indey >=460) & (indey < 480):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-460)*IMAGE_SIZE, 23*IMAGE_SIZE))
                elif (indey >=480) & (indey < 500):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-480)*IMAGE_SIZE, 24*IMAGE_SIZE))
                elif (indey >=500) & (indey < 520):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-500)*IMAGE_SIZE, 25*IMAGE_SIZE))
                elif (indey >=520) & (indey < 540):
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-520)*IMAGE_SIZE, 26*IMAGE_SIZE))
                else:
                    from_image = Image.open(IMAGES_PATH + str(indey) + '.jpg')
                    to_image.paste(from_image, ((indey-540)*IMAGE_SIZE, 27*IMAGE_SIZE))

                indey = indey +1
            to_image = to_image.crop((0,0,weight,height))
            to_image.save(IMAGE_SAVE_PATH) # 保存新图

        
            a = cv2.imread(args['cache'] + '/imgout/' + str(nsc) + '.jpg')

            b = cv2.imread(args['cache'] + '/out2/' + str(nsc) + '.jpg')
            #kernel = np.ones((5,5),np.uint8)
            # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9),anchor=None)
            # bp = cv2.dilate(b,kernel,iterations=1)
            # ap = cv2.erode(a,kernel,iterations=1)


            im=cv2.add(a,b)
            
            cv2.imwrite(args['output']+str(nsc) + '.jpg', im)
            print('处理完成：' + str(nsc) + '.jpg')
            t3 = datetime.datetime.now().microsecond
            t4 = time.mktime(datetime.datetime.now().timetuple())
            strTime =((t4 - t2) * 1000 + (t3 - t1) / 1000)
            print(str(strTime) + "ms")

        
        
        
            nsc = nsc + 1
            time.sleep((args['time']*3600000 - strTime)/1000)
        deldir(args['cache'] + '/out/')
        deldir(args['cache'] + '/out1/')
        deldir(args['cache'] + '/out2/')
        deldir(args['cache'] + '/imgout/')
        deldir(args['cache'] + '/imgout1/')
        print('删除缓存成功')

        
        

