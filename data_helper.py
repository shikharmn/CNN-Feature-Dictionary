from tqdm import tqdm
from PIL import Image
from glob import glob
from skimage.transform import resize

import numpy as np
import os


def give_train(set_data, N):
    train_path = "./" + set_data + "/train/good/*.png"
    files = glob(train_path)
    dim = np.asarray(Image.open(files[0])).shape[0]

    data = np.array([resize(np.array(Image.open(i)), (dim // 2, dim // 2)) for i in tqdm(files[:N])])
    data = data.reshape((-1, dim // 2, dim // 2, 3))

    return data


def give_test(set_data, defect):
    
    anomaly_path = "./" + set_data + "/test/" + defect + "/*.png"
    mask_path = "./" + set_data + "/test/" + defect + "/*.png"
    files = glob(train_path)
    dim = np.asarray(Image.open(files[0])).shape[0]

    data = np.array([resize(np.array(Image.open(i)), (dim // 2, dim // 2)) for i in tqdm(files[:N])])
    data = data.reshape((-1, dim // 2, dim // 2, 3))

    return anomaly, mask


def get_patches(size, stride, data):
    dim, dim = (data[0]).shape[:-1]
    print(dim)
    image_patches = np.array([[[[img[y:y + size, x:x + size]]
                                for x in range(0, (dim - size + 1), stride)]
                               for y in range(0, (dim - size + 1), stride)]
                              for img in data]
                             )
    #    for img in data:
    #        for y in range(0, (dim - size + 1) // stride, stride):
    #            for x in range(0, (dim - size + 1) // stride, stride):
    #                image_patches = np.append(image_patches, img[y:y + size, x:x + size])

    image_patches = image_patches.reshape((-1, size, size, 3))

    return (image_patches*256).astype(np.uint8)


# TESTING HELPERS

if __name__ == '__main__':
    train_data = give_train("wood", 40)
    test_data_anomaly, test_data_mask = give_test("wood", "hole")
    print(train_data.shape, test_data_anomaly.shape, test_data_mask.shape)
