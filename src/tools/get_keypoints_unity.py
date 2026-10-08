import os
import cv2
import numpy as np
import pycocotools.coco as coco
from multiprocessing import Pool


mfile_train = open('train.txt', 'w')
#mfile_val = open('test.txt', 'w')
m_coco = coco.COCO('annotations/kitti_train.json')
def _get_rect_and_mask(image):
    alpha = image[:, :, 3].copy()
    alpha[alpha < 200] = 0
    alpha[alpha >=200] = 255
    w_mean = np.sum(alpha, 0)/alpha.shape[0]
    h_mean = np.sum(alpha, 1)/alpha.shape[1]
    w_min = -1
    w_max = -1
    h_min = -1
    h_max = -1
    for i in range(0, alpha.shape[1]):
        if abs(w_mean[i] - 0) > 0.1:
            if w_min == -1:
                w_min = i
            else:
                w_max = i
        for i in range(0, alpha.shape[0]):
            if abs(h_mean[i] - 0) > 0.1:
                if h_min == -1:
                    h_min = i
                else:
                    h_max = i
    return [w_min, h_min, w_max, h_max], alpha


def get_new_image(name):
    global m_coco
    image = cv2.imread('train/image/' + name, -1)
    print(name)
    image_id = int(name.split('.')[0])
    ann_ids = m_coco.getAnnIds(imgIds=[image_id])
    anns = m_coco.loadAnns(ids=ann_ids)
    num_objs = len(anns)
    line = name
    for i in range(num_objs):
        ann = anns[i]
        keypoints = ann['keypoints']
#         new_path = os.path.join('train/new_image/', name.split('.')[0] + '_' + str(i) + '.png')
        new_path = os.path.join('train/new_image/', name.split('.')[0] + '.png')
#         bbox, alpha = _get_rect_and_mask(image)
        bbox = ann['bbox']
        width = bbox[2]
        height = bbox[3]
        xmin,ymin = bbox[:2]
        xmax, ymax = bbox[0] + width, bbox[1] + height
        xmin = max(0, int(xmin - width * 0.15))
        ymin = max(0, int(ymin - height * 0.15))
        xmax = min(image.shape[1], int(xmax + width * 0.15))
        ymax = min(image.shape[0], int(ymax + height * 0.15))
        print(bbox)
        new_image = image[ymin:ymax, xmin:xmax]
        for j in range(9):
            keypoints[j * 3] = keypoints[j * 3] - xmin
            keypoints[j * 3 + 1] = keypoints[j * 3 + 1] - ymin
            line = line + ',' + str(keypoints[j * 3]) + ',' + str(keypoints[j * 3 + 1])
        cv2.imwrite(new_path, new_image)
    line = line + '\n'
    return line

if __name__=="__main__":
    p = Pool(20)
    mlist = os.listdir('train/image')
    results = []
    for data in mlist:
        results.append(p.apply_async(get_new_image, args=(data,)))
    p.close()
    for result in results:
        line = result.get()
        print(line)
        mfile_train.write(line)
    
