import cv2
import numpy as np
import glob
from shutil import copyfile

if __name__ == "__main__":
     imlist = glob.glob("./output_whitebg/*.jpg")
     assert len(imlist) != 0
     for img in imlist: 
        im = cv2.imread(img)
        im_name = (img.split("/"))[-1]
        im_gazebo = cv2.imread("./gazebo_images/" + im_name)
        im_gazebo = im_gazebo[200:316+690, 100:350+486]
        copyfile(img, "./crane_filtered/trainA/" + im_name)
        cv2.imwrite("./crane_filtered/" + im_name, im_gazebo)
        im_show = np.hstack((im, im_gazebo))
        cv2.imshow("output", im_show)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     cv2.destroyAllWindows()
