import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import collections
import cv2
from nibabel.affines import apply_affine, voxel_sizes
import numpy.linalg as npl


def msseg():
    base_dir = "/home/huahong/Documents/Datasets/MSSEG_2016/Pre-processed training dataset/"
    all_res = {}
    for sub_dir in os.listdir(base_dir)[0:1]:
        if not os.path.isdir(os.path.join(base_dir, sub_dir)):
            continue
        all_res[sub_dir] = {}
        for file in os.listdir(os.path.join(base_dir, sub_dir)):
            if not 'preprocessed.nii.gz' in file:
                continue
            img = nib.load(os.path.join(base_dir, sub_dir, file))
            print(img.affine)
            # data = img.get_fdata()
            # print(data.shape)
            # print(npl.inv(img.affine))
    #         all_res[sub_dir][file] = str(img.header)
    #         # print(img.header)
    # values_set = collections.defaultdict(set)
    # for k in sorted(all_res.keys()):
    #     for i in all_res[k]:
    #         cur_value = all_res[k][i].splitlines()[42]
    #         values_set[cur_value].add(k)
    #     # _, v = cur_value.split(':')
    #     # v = v[2:-1].split(' ')
    #     # v = '[' + ' '.join(['%0.2f' % float(j) for j in v if j]) + ']'
    #     # print(k, v)
    # # for v in values_set:
    # #     print(v, values_set[v])


def isbi():
    all_res = {}
    base_dir = "/home/huahong/Documents/Datasets/isbi_dataset/raw"
    for i in range(1, 6):
        for j in range(1, 6):
            for modality in ['t1', 'flair', 't2', 'pd']:
                path = os.path.join(base_dir, 'training_%02d_%02d_%s.nii' % (i, j, modality))
                if not os.path.exists(path):
                    continue
                img = nib.load(path)
                all_res[(i, j, modality)] = str(img.header)
    for i in range(40, 45):
        for k in sorted(all_res.keys()):
            print(k, all_res[k].splitlines()[i])
        print("--------------------------------------")


def test_axes():
    path = '/home/huahong/Documents/Datasets/MSSEG_2016/Pre-processed training dataset/01016SACH/FLAIR_preprocessed.nii.gz'
    img = nib.load(path)
    affine = img.affine
    print(affine)
    voxel_sizes = nib.affines.voxel_sizes(affine)
    ratio = [k for i, k in enumerate(voxel_sizes) if i != 1]
    ratio = ratio[1] / ratio[0]
    data = img.get_fdata()[:, 200, :]
    height, width = data.shape[:2]

    if ratio > 1:
        height, width = height, int(width * ratio)
    else:
        height, width = int(height / ratio), width
    data = cv2.resize(data, (width, height), cv2.INTER_LINEAR)
    plt.imshow(data)
    plt.show()


if __name__ == '__main__':
    # msseg()
    # isbi()
    test_axes()