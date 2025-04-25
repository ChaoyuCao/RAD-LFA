import cv2
import pickle
import os
import numpy as np


# read all images in a dir
def image_loader(dir):
    images = []
    entries = os.listdir(dir)
    names = []
    for entry in entries:
        image = cv2.imread(os.path.join(dir, entry))
        images.append(image)
        names.append(entry)
    return images, names


# show image
def img_show(image, winname='image', wait=True, flag=cv2.WINDOW_NORMAL):
    cv2.namedWindow(winname, flag)
    cv2.imshow(winname, image)
    if wait:
        cv2.waitKey(0)


# save and load pkl file
def pkl_saver(path, data):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


# read pkl file
def pkl_loader(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    return data


# check dir exists
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# transfer minutes to seconds
def min_to_sec(time_point):
    """
    Minute --> second
    :param time_point: string, e.g., '5.15'
    :return: time in second
    """
    minute = int(time_point.split('.')[0])
    second = int(time_point.split('.')[-1])
    return minute * 60 + second


# calculate R2 in curve fitting
def cal_r2(x, y, func, popt):
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


# concentration setting tool
concentrations = {'HCG': [0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 
                          200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]}


def congrad(task, dup=3):
    cons = concentrations[task]
    cons_dup = []
    for con in cons:
        cons_dup += [con] * 3
    return {'concentration': cons_dup,
            'length': len(cons)}



