import openslide
import numpy as np
# import cv2
import PIL.Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def readsvs():
    svs_dir = './svs'
    # svs_path_list = []
    svs_name_list = []
    for i in os.listdir(svs_dir):
        path_name = os.path.join(i)
        # path_svs = os.path.join(svs_dir,i)
        svs_name_list.append(path_name)
    return svs_name_list  # S1700028Y.svs


def main(svs_name):
    # test = openslide.open_slide('./svs/S1700028Y.svs')
    svs_dir = './svs/'
    test = openslide.open_slide(svs_dir + svs_name)
    img = np.array(test.read_region((0, 0), 0, test.dimensions))
    output_path = r'./outputpng'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    png_name = svs_name.split('.')[0]
    PIL.Image.fromarray(img).save('./outputpng/' + png_name + '.png')


if __name__ == '__main__':
    svs_name_list = readsvs()
    for i in range(len(svs_name_list)):
        # for i in range(2):
        abc = main(svs_name_list[i])




