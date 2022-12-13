import os
import cv2
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from nets.makedir import mkdir
from nets.deldir import deldir

from utils.utils_metrics import compute_mIoU, show_results
Image.MAX_IMAGE_PIXELS = 2300000000
num_classes     = 2
name_classes    = ["background","root"]
VOCdevkit_path  = 'VOCdevkit'
image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines() 
miou_out_path   = "miou_out" 
path = "VOCdevkit/VOC2007/out2/"
gt_dir="miou_out/imgout/"
segpath="VOCdevkit/VOC2007/seg/"
pred_dir="miou_out/seg/"
mkdir(path)
mkdir(segpath)
mkdir(gt_dir)
mkdir(pred_dir)


f = os.listdir(path)
n = 0
i = 0

for i in f:
    img = cv2.imread(path + f[n]) # 填要转换的图片存储地址
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img[img<127]=0
    img[img>=127]=1
    
    a = f[n].strip('.jpg')
    print(a)
    cv2.imwrite(gt_dir + a + '.png',img) # 填转换后的图片存储地址，若在同一目录，则注意不要重名
    n=n+1


f1 = os.listdir(segpath)
n1 = 0
i1 = 0
for i1 in f1:
    img1 = cv2.imread(segpath + f1[n1]) # 填要转换的图片存储地址
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1[img1<127]=0
    img1[img1>=127]=1
    cv2.imwrite(pred_dir + f1[n1],img1) # 填转换后的图片存储地址，若在同一目录，则注意不要重名
    n1=n1+1
    

hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)