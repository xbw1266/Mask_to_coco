import json
import os
import glob
import random
import cv2
from tqdm import tqdm
	
    
def readJson(json_file):
    assert json_file.endswith(".json"), "Wrong json file input!"
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def changeBg(mask, src, bg):
    bg = bg[:806, :736]
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask_gray, 0,  1, cv2.THRESH_BINARY)
    # print(thresh.shape)
    # print(src.shape)
    # print(bg.shape)
    src[thresh==0] = bg[thresh == 0]
    return src

construction_dic = {1: "1.png", 2: "2.png", 3: "3.jpg"}

if __name__ == "__main__":
    data = readJson("good_img.json")
    for img_name in tqdm(data['good']):
        mask = cv2.imread("../data/mask_all1/" + img_name + ".png")
        mask = mask[200:316+690, 100:350+486]
        rgb_whitebg = cv2.imread("../data/output_whitebg/" + img_name + ".jpg")
        assert mask.shape == rgb_whitebg.shape, "Image shapes do not equal"
        construction_img = construction_dic[random.randint(1, 3)]
        construction_img = cv2.imread("../data/construction_bg/" + construction_img)
        assert construction_img is not None
        output = changeBg(mask, rgb_whitebg, construction_img)
        # cv2.imshow("Output", output)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break
        cv2.imwrite("../swapped_bg_cropped/" + img_name + ".jpg", output)