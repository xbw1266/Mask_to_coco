import base64
import json
import os
import glob
import cv2
import numpy as np
import functools
import time
from tqdm import tqdm

def timer(func):
	@functools.wraps(func)
	def wrapper_timer(*args, **kwargs):
		start_time = time.time()
		value = func(*args, **kwargs)
		end_time = time.time()
		run_time = end_time - start_time
		print("Finished {} in {} secs".format(func.__name__, run_time))
		return value
	return wrapper_timer 	

# define global variables:
root = "./data"
all_mask_dir = os.path.join(root, "mask_all1")
all_mask_filtered_dir = os.path.join(root, "mask_all1_filtered")
cropped_mask_black = os.path.join(root, "output")
cropped_mask_white = os.path.join(root, "output_whitebg")
cropped_swapped_bg = 'swapped_bg_cropped'

# json file:
root_json = "train.json"
def getRootjson(impath):
    im_id = 0
    data = {'images':[]}
    for img_name in glob.glob(impath + "/*.jpg"):
        print(img_name)
        im = cv2.imread(img_name)
        img_local_name = img_name.split("/")[-1]
        im_info = {'height': im.shape[0], 'width': im.shape[1], 'id': im_id, 'file_name': img_local_name}
        data['images'].append(im_info)
        im_id += 1
    with open(root_json, 'w') as f:
        json.dump(data, f, indent=4)


def getImgInfo(img, img_path):
    assert img is not None, "Image data error!"
    data = {'version' : '4.2.9', 'flags' : {}, 'shapes' : [{'label' : 'baxter', 'points' : [], 'group_id' : None, 'shape_type' : 'polygon', 'flags' : {}}], 'imagePath' : '', 'imageData' : '', 'imageHeight' : '', 'imageWidth' : ''}
    corners = getCorners(img, 15, False)
    if corners is not None:
        for point in corners:
            data['shapes'][0]['points'].append([int(point[0]), int(point[1])])
        data['imagePath'] = img_path.split('/')[-1]
        img_name_no_ext = data['imagePath'].split('.')[0]
        with open(img_path, 'rb') as f:
            data['imageData'] = str(base64.b64encode(f.read()))
        data['imageHeight'] = img.shape[1]
        data['imageWidth'] = img.shape[0]
        with open(os.path.join("img_json", img_name_no_ext + ".json"), 'w+') as f:
            json.dump(data, f, indent=4)
        return True
    else:
        return False

# using Harris corner detection to find corners?
def getCorners(img, sample_rate=10, show_image=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 30, 200)
    cnts, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    output = sort_points(cnts)
    # print(output.shape)
    sub_sampled = output[0::sample_rate]
    if show_image:
        for point in sub_sampled:
            print(point)
            cv2.circle(img, tuple(point), 3, (0, 0, 255), 1)
        cv2.imshow("ddd", img)
        cv2.waitKey(30)
    return sub_sampled

def sort_points(cnts):
    output = np.squeeze(cnts[0])
    for cnt in cnts[1:]:
        new_cnt = np.squeeze(cnt)
        output = np.vstack((output, new_cnt))
    return output
        

def get_corners_from_contours(contours, corner_amount=16):
    while True:
        # epsilon = coefficient * cv2.arcLength(contours, True) if coefficient > 0 else 0
        epsilon = 0.001
        # print("epsilon:", epsilon)
        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        return hull

def validation(img_folder):
    pass


def readJson(json_file):
    assert json_file.endswith(".json"), "Wrong json file input!"
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def rawSorting(img_path):
    bad_img_list = []
    img = glob.glob(img_path + "/*.png")
    for img_path in tqdm(img):
        img_cv = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        img_name_no_ext = img_name.split(".")[0]
        img_cv = img_cv[200:316+690, 100:350+486]
        save_loc = "./output"
        if getImgInfo(img_cv, img_path):
            cv2.imwrite(os.path.join(save_loc, img_name_no_ext + ".jpg"), img_cv)
        else:
            bad_img_list.append(img_path)
    print("{} images processed, {} failed.".format(len(img), len(bad_img_list)))
    print("Bad images are:")
    print(bad_img_list)

def getPolyInfo(mask_img, swapped_bg_cropped_path):
    assert mask_img is not None, "Image data error!"
    data = {'version' : '4.2.9', 'flags' : {}, 'shapes' : [{'label' : 'baxter', 'points' : [], 'group_id' : None, 'shape_type' : 'polygon', 'flags' : {}}], 'imagePath' : '', 'imageData' : '', 'imageHeight' : '', 'imageWidth' : ''}
    corners = getCorners(mask_img, 15, False)
    if corners is not None:
        for point in corners:
            data['shapes'][0]['points'].append([int(point[0]), int(point[1])])
        data['imagePath'] = swapped_bg_cropped_path.split('/')[-1]
        img_name_no_ext = data['imagePath'].split('.')[0]
        with open(swapped_bg_cropped_path, 'rb') as f:
            data['imageData'] = str(base64.b64encode(f.read()))
        data['imageHeight'] = mask_img.shape[1]
        data['imageWidth'] = mask_img.shape[0]
        with open(os.path.join(cropped_swapped_bg, img_name_no_ext + ".json"), 'w+') as f:
            json.dump(data, f, indent=4)
        return True
    else:
        return False

def generatePolygonJson(good_json, img_path):
    data = readJson(good_json)
    bad_list = []
    for img_name in tqdm(data['good']):
        mask = cv2.imread(os.path.join(all_mask_dir, img_name + ".png"))
        assert mask is not None
        mask = mask[200:316+690, 100:350+486]
        if not getPolyInfo(mask, os.path.join(img_path, img_name + ".jpg")):
            bad_list.append(img_name)
    print("{} images processed, {} failed.".format(len(data['good']), len(bad_list)))
    print("Bad images are:")
    print(bad_list)
            
        
    

if __name__ == "__main__":
    #getRootjson(cropped_swapped_bg)
    generatePolygonJson("good_img.json", cropped_swapped_bg)
    