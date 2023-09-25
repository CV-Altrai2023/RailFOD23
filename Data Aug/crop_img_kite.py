import os
import shutil
from os.path import join
import cv2
import glob

root_dir = "./images/train2017/"
save_dir = "./crop_img"

jpg_list = glob.glob(root_dir + "/*.jpg")
print(jpg_list)

fo = open("dpj_small.txt", "w")

max_s = -1
min_s = 1000

for jpg_path in jpg_list:
    # jpg_path = jpg_list[3]
    label1 = jpg_path.replace('images', 'labels')
    txt_path = label1.replace("jpg", "txt")
    jpg_name = os.path.basename(jpg_path)

    f = open(txt_path, "r")

    img = cv2.imread(jpg_path)

    height, width, channel = img.shape
    try:
        file_contents = f.readlines()
        for num, file_content in enumerate(file_contents):
            # print(num)
            clss, xc, yc, w, h = file_content.split()
            if clss =='33':
                print('kite ok')
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)

                xc *= width
                yc *= height
                w *= width
                h *= height

                max_s = max(w*h, max_s)
                min_s = min(w*h, min_s)

                half_w, half_h = w // 2, h // 2

                x1, y1 = int(xc - half_w), int(yc - half_h)
                x2, y2 = int(xc + half_w), int(yc + half_h)

                crop_img = img[y1:y2, x1:x2]
                if w>=30 or h>=30:
                    new_jpg_name = jpg_name.split('.')[0] + "_crop_" + str(num) + ".jpg"
                    cv2.imwrite(os.path.join(save_dir, new_jpg_name), crop_img)
                # cv2.imshow("croped",crop_img)
                # cv2.waitKey(0)
                fo.write(os.path.join(save_dir, new_jpg_name)+"\n")
    except:
        fo.write("end \n")
    f.close()

fo.close()

print(max_s, min_s)