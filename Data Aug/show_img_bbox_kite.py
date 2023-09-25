import glob
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)

init_n = 0
def vis_result(file_contents, jpg_path, save_path):
    global init_n
    img = cv2.imread(jpg_path)

    f = open(file_contents,"r")

    height, width, _ = img.shape

    f_c = f.readlines()
    if not os.path.exists("./visresult/"+str(init_n)+'/'):
        os.makedirs("./visresult/"+str(init_n)+'/')
    for line in f_c:
        clss, xc, yc, w, h = line.split()
        # print(clss)
        if clss ==str(init_n):
            print(str(init_n)+'ok')
            xc, yc, w, h = float(xc), float(yc), float(w), float(h)

            xc *= width
            yc *= height
            w *= width
            h *= height

            half_w, half_h = w // 2, h // 2
            x1, y1 = int(xc - half_w), int(yc - half_h)
            x2, y2 = int(xc + half_w), int(yc + half_h)

            c = [x1,y1,x2,y2]

            plot_one_box(c, img)

            newname = os.path.basename(file_contents).split('.')[0] + "_vis.jpg"
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join("./visresult/"+str(init_n)+'/', newname), img)
            init_n+=1


if __name__ == "__main__":

    import shutil
    save_path = "./visresult/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    # vis processed imgs
    vis_path = "./images/train2017/"

    vis_img_list = glob.glob(vis_path+"/*.jpg")

    for c in vis_img_list:
        label1 = c.replace('images', 'labels')
        label = label1.replace('jpg', 'txt')
        vis_result(label, c, save_path)