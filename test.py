import os
from time import sleep

import numpy as np
from ImagesCameras import ImageTensor

path = "/home/godeta/Images/ICCV/Data_publi/Lynred_day/vis/"
img_list = [path + file for file in os.listdir(path)]
idx = np.random.randint(0, len(img_list), size=8)
id = idx[0]
img = ImageTensor(img_list[id]).batch([ImageTensor(img_list[i]) for i in idx[1:]])
screen = img.show(name='img', opencv=True, asyncr=True)
for i in range(5):
    idx = np.random.randint(0, len(img_list), size=8)
    id = idx[0]
    img = ImageTensor(img_list[id]).batch([ImageTensor(img_list[i]) for i in idx[1:]])
    screen.update(img)
    print()
    sleep(5)
