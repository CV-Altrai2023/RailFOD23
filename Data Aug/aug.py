# import glob
import cv2 as cv2
import numpy as np
# from PIL import Image
import random
import math
from os.path import basename, split, join, dirname
from util import *


def find_str(filename):
    if 'train' in filename:
        return dirname(filename[filename.find('train'):])
    else:
        return dirname(filename[filename.find('val'):])


def convert_all_boxes(shape, anno_infos, yolo_label_txt_dir):
    height, width, n = shape
    label_file = open(yolo_label_txt_dir, 'w')
    for anno_info in anno_infos:
        target_id, x1, y1, x2, y2 = anno_info
        b = (float(x1), float(x2), float(y1), float(y2))
        bb = convert((width, height), b)
        label_file.write(
            str(target_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def save_crop_image(save_crop_base_dir, image_dir, idx, roi):
    crop_save_dir = join(save_crop_base_dir, find_str(image_dir))
    check_dir(crop_save_dir)
    crop_img_save_dir = join(
        crop_save_dir,
        basename(image_dir)[:-3] + '_crop_' + str(idx) + '.jpg')
    cv2.imwrite(crop_img_save_dir, roi)


def GaussianBlurImg(image):
    # 高斯模糊
    ran = random.randint(0, 9)
    if ran % 2 == 1:
        image = cv2.GaussianBlur(image, ksize=(ran, ran), sigmaX=0, sigmaY=0)
    else:
        pass
    return image


def roi_resize(image, h, w):
    # 改变图片大小
    height, width, channels = image.shape

    # while (height * width) > area_max:
    #     image = cv2.resize(image, (int(width * 0.9), int(height * 0.9)))
    #     height, width, channels = image.shape
    #     height, width = int(height * 0.9), int(width * 0.9)
    #
    # while (height * width) < area_min:
    #     image = cv2.resize(image, (int(width * 1.1), int(height * 1.1)))
    #     height, width, channels = image.shape
    #     height, width = int(height * 1.1), int(width * 1.1)

    image = cv2.resize(image,None,fx=h,fy=w)   # 注意，目标size不能太大，否则图片会不够大小贴下目标

    return image


def copysmallobjects(image_dir, label_dir, save_base_dir, small_img_dir,
                      times):
    image = cv2.imread(image_dir)
    rows1, cols1, channels1 = image.shape
    out_mask = np.zeros((rows1, cols1, 1), np.uint8)
    # labels = read_label_txt(label_dir)
    # if len(labels) == 0:
    #     return

    # yolo txt转化为x1y1x2y2
    # rescale_labels = rescale_yolo_labels(labels, image.shape)  # 转换坐标表示
    # print("org bbox:", rescale_labels)  # 原图像bbox集合
    rescale_labels=[]
    all_boxes = []

    # for _, rescale_label in enumerate(rescale_labels):
    #     all_boxes.append(rescale_label)

    for small_img_dirs in small_img_dir:
        h_b, w_b, c_b = image.shape
        image_bbox = cv2.imread(small_img_dirs)
        # from 3000 to 1500
        h_r, _,_ = image_bbox.shape
        scale_r = h_b/(10*h_r)
        # scale_r = 1
        roi = roi_resize(image_bbox, scale_r, scale_r)  # 对roi图像做缩放

        label_path1 = small_img_dirs.replace('have object', 'object label')
        label_path = label_path1[:-4]+'.png'
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_copy = label.copy().astype('int32')
        unique_ray = np.unique(label_copy)
        label_copy[label == unique_ray[1]] = 1
        if unique_ray[1]==75:
            class_obj=1
        elif unique_ray[1]==14:
            class_obj=2
        elif unique_ray[1]==38:
            class_obj=3
        elif unique_ray[1]==113:
            class_obj=4
        # label_copy[label == 7] = 1
        # label_copy[label == 10] = 0
        # img=cv2.imread('./qqq.png')
        dilate=cv2.resize(label_copy, None, fx=scale_r, fy=scale_r, interpolation = cv2.INTER_NEAREST)
        # print('===', rescale_labels)
        rows, cols = dilate.shape
        new_bboxes = random_add_patches(roi.shape,     # 此函数roi目标贴到原图像上，返回的bbox为roi在原图上的bbox,
                                         rescale_labels,  # 并且bbox不会挡住图片上原有的目标
                                         image.shape,
                                         paste_number=1,  # 将该roi目标复制几次并贴到到原图上
                                         iou_thresh=0.5,
                                        cl=class_obj)    # iou_thresh 原图上的bbox和贴上去的roi的bbox的阈值
        print(new_bboxes)
        count = 0
        # print("end patch")
        for new_bbox in new_bboxes:
            count += 1

            cl, bbox_left, bbox_top, bbox_right, bbox_bottom = new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3], \
                                                               new_bbox[4]
            #roi = GaussianBlurImg(roi)  # 高斯模糊
            height, width, channels = roi.shape
            center = (int(width / 2), int(height / 2))
            #ran_point = (int((bbox_top+bbox_bottom)/2),int((bbox_left+bbox_right)/2))
            mask = 255 * np.ones(roi.shape, roi.dtype)
            # print("before try")
            try:
                if count > 1:  # 如果count>1,说明paste_number大于1次，对roi做一个翻转变换
                    roi = flip_bbox(roi)
                #image[bbox_top:bbox_bottom, bbox_left:bbox_right] = roi
                #image[bbox_top:bbox_bottom, bbox_left:bbox_right] = cv2.addWeighted(image[bbox_top:bbox_bottom, bbox_left:bbox_right],
                #                                                                    0.5,roi,0.5,0) #图片融合

                # 融合 cv2.seamlessClone
                #image = cv2.seamlessClone(roi, image, mask, ran_point, cv2.NORMAL_CLONE)
                #print(str(bbox_bottom-bbox_top) + "|" + str(bbox_right-bbox_left))
                #print(roi.shape)
                #print(mask.shape)
                # image[bbox_top:bbox_bottom, bbox_left:
                #       bbox_right] = cv2.seamlessClone(
                #           roi,
                #           image[bbox_top:bbox_bottom, bbox_left:bbox_right],
                #           mask, center, cv2.NORMAL_CLONE)

                center=[bbox_top,bbox_left]#在新背景图片中的位置
                for i in range(rows):
                    for j in range(cols):
                        if dilate[i,j]==1:#0代表黑色的点
                            image[center[0]+i,center[1]+j]=roi[i,j]#此处替换颜色，为BGR通道
                            out_mask[center[0] + i, center[1] + j] = 1
                all_boxes.append(new_bbox)
                # rescale_labels.append(new_bbox)

                # print("end try")
            except ValueError:
                print("---")
                continue
    # print("end for")
    dir_name = find_str(image_dir)
    save_dir = join(save_base_dir, dir_name)
    check_dir(save_dir)
    yolo_txt_dir = join(
        save_dir+'/labels/',
        basename(image_dir.replace('.jpg', '_aug_%s.txt' % str(times))))
    cv2.imwrite(
        join(save_dir+'/images/',
             basename(image_dir).replace('.jpg', '_aug_%s.jpg' % str(times))),
        image)
    cv2.imwrite(
        join(save_dir+'/mask/',
             basename(image_dir).replace('.jpg', '_aug_%s.png' % str(times))),
        out_mask)
    convert_all_boxes(image.shape, all_boxes, yolo_txt_dir)
