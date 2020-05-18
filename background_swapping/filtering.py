import os
import glob
import json

image_dir = "../data/whitebg_filtered/*.jpg"

if __name__ == "__main__":
	data = {"good" : []}
	for img in glob.glob(image_dir):
		print(img)
		image_name = img.split("/")[-1]
		image_name_no_ext = image_name.split(".")[0]
		data["good"].append(image_name_no_ext)
	
	with open("good_img.json", "w+") as f:
		json.dump(data, f, indent=4)
