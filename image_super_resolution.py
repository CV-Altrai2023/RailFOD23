import mmcv
import matplotlib.pyplot as plt
from mmagic.apis import MMagicInferencer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Create a MMagicInferencer instance and infer
editor = MMagicInferencer('esrgan')

# 指定图片文件夹的路径
image_folder = './aigc_ours'

# 获取文件夹中的所有JPG文件
jpg_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
result_out_dir = './super_img'


for img_name in jpg_files:
	results = editor.infer(img=os.path.join(image_folder,img_name),
	                       result_out_dir=os.path.join(result_out_dir,img_name))
