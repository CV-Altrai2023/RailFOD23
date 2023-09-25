import json
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# COCO数据集的标注文件路径
annotation_file = 'coco_aigc.json'  # 将路径替换为你的标注文件路径

# 初始化COCO API
coco = COCO(annotation_file)

# 获取所有图像的ID
image_ids = coco.getImgIds()
print(len(image_ids))

# 循环遍历图像并可视化
for image_id in image_ids:
	# 获取图像信息
	image_info = coco.loadImgs(image_id)[0]

	# 加载图像
	image = cv2.imread('RailFOD23/Images/' + image_info['file_name'])  # 将路径替换为你的图像文件夹路径
	print(image_info['file_name'])



	# 获取图像的标注信息
	ann_ids = coco.getAnnIds(imgIds=image_id)
	annotations = coco.loadAnns(ann_ids)

	# 绘制图像
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

	# 绘制标注信息
	for annotation in annotations:
		category_id = annotation['category_id']
		category_info = coco.loadCats(category_id)[0]
		category_name = category_info['name']

		bbox = annotation['bbox']
		x, y, width, height = bbox

		# 创建矩形框
		rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')

		# 添加矩形框到图像
		plt.gca().add_patch(rect)

		# 添加类别标签
		plt.text(x, y - 5, category_name, color='r', fontsize=12)

	# 显示图像
	plt.axis('off')
	plt.show()
