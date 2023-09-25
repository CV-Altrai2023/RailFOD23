from mmagic.apis import MMagicInferencer


file_path = "text4.txt"  # 替换为你的文件路径
out_path = "output%s"%(file_path.split(".")[0][-1])
text=[]
try:
	with open(file_path, "r") as file:
		for line in file:
			if len(line) == 1:
				print("空行")
				pass
			else:
				text.append(line.strip())
except FileNotFoundError:
	print("文件未找到")
except Exception as e:
	print("发生错误:", e)

# Create a MMagicInferencer instance and infer
sd_inferencer = MMagicInferencer(model_name='stable_diffusion')
for i in range(len(text)):
	result_out_dir = out_path + '/%s.jpg' % i
	results = sd_inferencer.infer(text=text[i], result_out_dir=result_out_dir)