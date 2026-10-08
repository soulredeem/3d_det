
import cv2
import numpy as np
from PIL import Image
import os, glob
def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    image = np.ascontiguousarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1]

def draw_box_3d(image, corners, c=(0, 0, 255)):
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (int(corners[f[j], 0]), int(corners[f[j], 1])),
                     (int(corners[f[(j + 1) % 4], 0]), int(corners[f[(j + 1) % 4], 1])), c, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (int(corners[f[0], 0]), int(corners[f[0], 1])),
                     (int(corners[f[2], 0]), int(corners[f[2], 1])), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (int(corners[f[1], 0]), int(corners[f[1], 1])),
                     (int(corners[f[3], 0]), int(corners[f[3], 1])), c, 1, lineType=cv2.LINE_AA)
    return image

if __name__ == "__main__":

    
    mfile = open('train_new.txt', 'r')
    nlist = []
    kpslist = []
    for line in mfile.readlines():
        line = line.strip()
        line = line.split(',')
        nlist.append(line[0])
        kpslist.append(line[1:])
    mfile = open('val_new.txt', 'r')
    for line in mfile.readlines():
        line = line.strip()
        line = line.split(',')
        nlist.append(line[0])
        kpslist.append(line[1:])
    # srcpath = './003'
    # srcpath = './all'
    srcpath = './003_done'
    imgs_per = glob.glob(os.path.join(srcpath, '*.*g'))
    for i, img_per in enumerate(imgs_per):
        if(i%(len(imgs_per)//20)==0):
            print('processing...   {:.2f}%'.format(i/len(imgs_per)*100))
        img_name = img_per.split('/')[-1]
        idx = nlist.index(img_name)
        img = read_image_bgr(img_per)
        kps = np.array(kpslist[idx]).reshape(9,2)
        kps = kps.astype(float)
        draw = img.copy()
        green = (0, 255, 0)

        draw = draw_box_3d(draw, kps[0:8], green)
        cv2.imwrite('./003_done_3d/{}'.format(img_name), draw)
        # break
    # kps = ['485.4996337890625', '255.71240234375','243.911376953125', '264.2855224609375', '65.54901123046875', '196.92364501953125', '233.5511474609375', 
    # '193.08306884765625', '485.49', '34.93', '243.911376953125', '33.66162109375', '65.54901123046875','43.674896240234375', '233.5511474609375',
    #  '44.24578857421875', '237.61474609375', '130.14552307128906']
    # print(type(img))
    # print(type(kps))
    
    # kps = np.array(kps).reshape(9,2)
    # kps = kps.astype(float)
    # print(img.dtype)
    # draw = img.copy()
    # green = (0, 255, 0)

    # draw = draw_box_3d(draw, kps[0:8], green)
    # cv2.imwrite('result.png', draw)draw_kp.py
