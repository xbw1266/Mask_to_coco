import cv2
import glob
from tqdm import tqdm
import numpy as np



# src: h: 806, w: 736
# bg : max h: 503 max w:400 
def swapBg(bg, src, mask):
    (w, h, c)  = bg.shape
    small = min(w, h)
    src_ratio = 806 // 736
    src_scaled = cv2.resize(src, (small, src_ratio* small), interpolation = cv2.INTER_CUBIC)
    (w1, h1, c) = src_scaled.shape
    mask_scaled = cv2.resize(mask, (w1, h1))
    assert w1 == h1, "w1 == h1"
    x, y = 0, 0
    for i in range(w1):
        for j in range(h1):
            pix = mask_scaled[j, i]
            if pix == 1:
                bg[y+j, x+i] = src_scaled[j, i]
    
    return bg

def getMask(src):
    mask = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(mask, 0,  1, cv2.THRESH_BINARY)
    #cv2.imshow("mask", thresh)
    #cv2.waitKey(0)
    return thresh
    
def getCrop(mask, src): 
    mask_out = cv2.subtract(mask, src)
    mask_out = cv2.subtract(mask, mask_out)
    return mask_out
    
if __name__ == "__main__":
    img_name = glob.glob("./all1/*.jpg")
    bg_name = glob.glob("./construction_site/*.jpg")
    i = 0
    for bg in bg_name: 
        bg_cv = cv2.imread(bg)
        for img in tqdm(img_name):
            name = img.split("/")[-1]
            mask_name = name.split(".")[0] + ".png"
            mask = "./mask_all1/" + mask_name
            mask = cv2.imread(mask)
            src = cv2.imread(img)
            output = getCrop( mask, src)
            output = output[200:316+690, 100:350+486]
            #cv2.imwrite("./output/" + name, output)
            th_mask = getMask(output)
            new_bg = bg_cv.copy()
            rtn = swapBg(new_bg, output, th_mask)
            #cv2.imshow("bg", rtn)
            #cv2.waitKey(1)
            save_name = str(i) + ".jpg"
            cv2.imwrite("./aug_output/" + save_name, rtn) 
            i += 1
                
     
                
        
