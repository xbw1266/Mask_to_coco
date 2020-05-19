import glob
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

test_dir = "baxter/test"
train_dir = "baxter/train"
root_dir = "baxter"

# define global variables:
root = "./data"
all_mask_dir = os.path.join(root, "mask_all1")
all_mask_filtered_dir = os.path.join(root, "mask_all1_filtered")
cropped_mask_black = os.path.join(root, "output")
cropped_mask_white = os.path.join(root, "output_whitebg")
cropped_swapped_bg = 'swapped_bg_cropped'


def getbbox(im_name):
    mask = cv2.imread(os.path.join(all_mask_dir, im_name + ".png"))[200:316+690, 100:350+486]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 0,  1, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    return x, y, w, h

def getseg(im_name):
    segmentation = []
    areas = []
    mask = cv2.imread(os.path.join(all_mask_dir, im_name + ".png"))[200:316+690, 100:350+486]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 0,  1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        contour = contour.flatten().tolist()
        # segmentation.append(contour)
        if len(contour) > 4:
            segmentation.append(contour)
            areas.append(area)
    assert len(segmentation) != 0
    max_index = areas.index(max(areas))
    return segmentation[max_index], areas[max_index]

# json file:
def getRootjson(impath, json_output):
    im_id = 0
    data = {'images':[], 'categories': [{'supercategory' : 'baxter', 'id': 0, 'name': 'baxter'}], 'annotations': []}
    for img_name in tqdm(glob.glob(impath + "/*.jpg")):
        img_name_no_ext = img_name.split("/")[-1]
        img_name_no_ext = img_name_no_ext.split(".")[0]
        label_info = []
        # print(img_name)
        im = cv2.imread(img_name)
        img_local_name = img_name.split("/")[-1]
        im_info = {'height': im.shape[0], 'width': im.shape[1], 'id': im_id, 'file_name': img_local_name}
        data['images'].append(im_info)
        
        segmentation_data, area = getseg(img_name_no_ext)
        # print(area)
        x, y, w, h = getbbox(img_name_no_ext)
        label_info ={
                "segmentation" : [segmentation_data],
                "area" : area,
                "iscrowd": 0,
                "image_id": im_id,
                "bbox": [x, y, w, h],
                "category_id": 0,
                "id": im_id + 1
            }
        im_id += 1
        data['annotations'].append(label_info)
    with open(json_output, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    getRootjson(test_dir, "baxter/test.json")
    # getRootjson(train_dir, "baxter/train.json")