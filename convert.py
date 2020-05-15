import base64
import json
import os
import glob
import cv2

# define global variables:
root = "./data"
all_mask_dir = os.path.join(root, "mask_all1")
cropped_mask_black = os.path.join(root, "output")
cropped_mask_white = os.path.join(root, "output_whitebg")

# json file:
root_json = "train.json"
def getRootjson():
    im_id = 0
    data = {'images':[]}
    for img_name in glob.glob(cropped_mask_white + "/*.jpg"):
        print(img_name)
        im = cv2.imread(img_name)
        img_local_name = img_name.split("/")[-1]
        im_info = {'height': im.shape[0], 'width': im.shape[1], 'id': im_id, ' file_name': img_local_name}
        data['images'].append(im_info)
        im_id += 1
    with open(root_json, 'w') as f:
        json.dump(data, f, indent=4)
        
def getImgInfo(img):
    assert img is not None, "Image data error!"
    data = {'version' : '4.2.9', 'flags' : {}, 'shapes' : [], 'imagePath', 'imageData', 'imageHeight', 'imageWidth'}
    

if __name__ == "__main__":
    getRootjson()

