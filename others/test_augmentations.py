from albumentations.augmentations.functional import grid_distortion, clahe
from albumentations.augmentations.domain_adaptation import fourier_domain_adaptation
from skimage.exposure import match_histograms
from imgaug import augmenters as iaa
import albumentations.augmentations.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import random
import cv2


def augmentations(data, ratio, trainSize):
    all_modalities = ['mask', 'flair', 'ref_flair']
    height, width = data[all_modalities[-1]].shape[:2]  # height/y for first axis, width/x for second axis

    # x = random.random()
    # print(x)
    # # for ratio around 1 and p=0.2, random assign a big ratio
    # if abs(1 - ratio) < 0.05 and x < 0.2:
    #     ratio = 1 + random.random()
    #     print('random assign ratio ', ratio)
    #     if random.random() < 0.5:
    #         ratio = 1 / ratio
    #
    # # p=0.8, reshape the image based on ratio
    # if abs(1 - ratio) > 0.05 and random.random() < 0.8:
    #     print('reshape')
    #     if ratio > 1:
    #         height, width = height, int(width * ratio)
    #     else:
    #         height, width = int(height / ratio), width
    #
    #     for modality in all_modalities:
    #         interpolation = cv2.INTER_LINEAR
    #         data[modality] = cv2.resize(data[modality], (width, height), interpolation)
    #
    # # p=0.3, randomly rotate angle between 1->45
    # if random.random() < 0:
    #     angle = random.randint(1, 45)
    #     matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    #     for modality in all_modalities:
    #         interpolation = cv2.INTER_LINEAR
    #         data[modality] = cv2.warpAffine(data[modality], M=matrix, dsize=(width, height), flags=interpolation)
    #
    # p=0.3, randomly distort image
    # if random.random() < 1:
    #     num_steps, distort_limit = 5, (-0.3, 0.3)
    #     xsteps = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in range(num_steps + 1)]
    #     ysteps = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in range(num_steps + 1)]
    #     for modality in all_modalities:
    #         data[modality] = grid_distortion(data[modality], num_steps, xsteps, ysteps,
    #                                          border_mode=cv2.BORDER_CONSTANT)
    #
    # p=0.5, do random crop and resize to trainSize
    need_resize = False
    if random.random() < 1:
        crop_size = random.randint(int(trainSize / 1.5), min(height, width))
        need_resize = True
    else:
        crop_size = trainSize

    if 'mask' not in data or np.sum(data['mask']) == 0 or random.random() < 0.005:
        x_min = random.randint(0, width - crop_size)
        y_min = random.randint(0, height - crop_size)
    else:
        mask = data['mask']
        non_zero_yx = np.argwhere(mask)
        y, x = random.choice(non_zero_yx)
        x_min = x - random.randint(0, crop_size - 1)
        y_min = y - random.randint(0, crop_size - 1)
        x_min = np.clip(x_min, 0, width - crop_size)
        y_min = np.clip(y_min, 0, height - crop_size)

    for modality in all_modalities:
        interpolation = cv2.INTER_LINEAR
        data[modality] = data[modality][y_min:y_min+crop_size, x_min:x_min+crop_size]
        if need_resize:
            data[modality] = cv2.resize(data[modality], (trainSize, trainSize), interpolation)
    height, width = trainSize, trainSize
    data['org_flair'] = data['flair'].copy()

    # # p=0.3, randomly flip
    # for axis in [1, 0]:
    #     x = random.random()
    #     print(axis, x, x < 0.3)
    #     if x < 0.3:
    #         for modality in all_modalities:
    #             data[modality] = np.flip(data[modality], axis).copy()
    #
    # # p=0.5, randomly transpose
    # x = random.random()
    # print(x, x < 0.5)
    # if x < 0.5:
    #     height, width = width, height
    #     for modality in all_modalities:
    #         transpose_index = (1, 0)
    #         data[modality] = np.transpose(data[modality], transpose_index)

    x = random.random()
    if x < 1:
        for modality in ['flair']:
            limit = random.uniform(1, 3)
            print(limit)
            max_value = np.max(data[modality])
            data_uint8 = (data[modality] / max_value * 255).astype(np.uint8)
            data[modality] = clahe(data_uint8, limit, (8, 8)).astype(np.float64) / 255 * max_value
    #
    # x = random.random()
    # print(x, x < 0.5)
    # if x < 1:
    #     for modality in ['flair']:
    #         brightness = random.uniform(0.65, 1.35)
    #         contrast = random.uniform(0.65, 1.35)
    #         transforms = [
    #             lambda x: F.adjust_brightness_torchvision(x, brightness),
    #             lambda x: F.adjust_contrast_torchvision(x, contrast),
    #         ]
    #         random.shuffle(transforms)
    #
    #         max_value = np.max(data[modality])
    #         data_uint8 = (data[modality] / max_value * 255).astype(np.uint8)
    #         for transform in transforms:
    #             data_uint8 = transform(data_uint8)
    #         data[modality] = data_uint8.astype(np.float64) / 255 * max_value

    # matched = match_histograms(data['flair'], data['ref_flair'])
    # blend_ratio = random.uniform(0.5, 1.0)
    # print(blend_ratio)
    # data['flair'] = cv2.addWeighted(matched, blend_ratio, data['flair'], 1 - blend_ratio, 0)

    # ksize = int(random.choice([3, 3, 3, 5, 5, 7]))
    # data['flair'] = F.blur(data['flair'], ksize)

    # data['flair'] = F.downscale(data['flair'], scale=0.7)

    # circle_w = random.randint(10, trainSize)
    # circle_h = min(random.randint(circle_w // 2, circle_w * 2), trainSize)
    # Y = np.linspace(-1, 1, circle_h)[:, None]
    # X = np.linspace(-1, 1, circle_w)[None, :]
    # gradient = X ** 2 + Y ** 2
    # gradient = np.clip(a=gradient, a_min=0.0, a_max=1.0)
    # gradient = 1 - gradient
    #
    # start_h = random.randint(0, trainSize - circle_h)
    # start_w = random.randint(0, trainSize - circle_w)
    # amp = random.uniform(-0.3, 0.3)
    # multiplier = np.ones((trainSize, trainSize))
    # multiplier[start_h:start_h + circle_h, start_w:start_w + circle_w] += gradient * amp
    # print(amp, circle_h, circle_w, start_h, start_w, np.max(multiplier), np.min(multiplier))

    # h, w = 128, 128
    # is_horizontal = random.random() < 0.5
    # multiplier = np.ones((h, w))
    # amp = random.uniform(-0.3, 0.3)
    # if is_horizontal:
    #     rec_h = random.randint(10, h)
    #     start_h = random.randint(0, h - rec_h)
    #     gradient = np.tile(1 - np.linspace(-1, 1, rec_h)**2, (w, 1)).T
    #     multiplier[start_h:start_h + rec_h] += gradient * amp
    # else:
    #     rec_w = random.randint(10, w)
    #     start_w = random.randint(0, h - rec_w)
    #     gradient = np.tile(1 - np.linspace(-1, 1, rec_w) ** 2, (h, 1))
    #     multiplier[:, start_w:start_w + rec_w] += gradient * amp
    # print(is_horizontal, amp)
    #
    # data['mult'] = multiplier
    # data['flair'] *= multiplier

    # data['flair'] = iaa.Sharpen(alpha=(0, 0.5), lightness=(0.5, 1.5)).augment_image(data['flair'])
    # sigma = random.uniform(0, 0.2)
    # mu = random.uniform(0, 0.1)
    # data['flair'] = np.random.normal(mu, sigma, (height, width)) + data['flair']
    # print(mu, sigma)
    if 'mask' in data:
        data['mask'] = (data['mask'] > 0.5).astype(np.float32)
    return data


fl_path = '/home/huahong/Documents/Datasets/isbi_dataset/raw/training_01_01_flair.nii'
mask_path = '/home/huahong/Documents/Datasets/isbi_dataset/raw/training_01_01_mask1.nii'
ref_flair_path = '/home/huahong/Documents/Datasets/msseg_dataset/raw/08031SEVE_12_01_flair.nii.gz'
fl_data = nib.load(fl_path).get_fdata()
fl_slice = np.rot90(fl_data[:,:,90])
mask_data = nib.load(mask_path).get_fdata()
mask_slice = np.rot90(mask_data[:,:,90])
ref_flair_data = nib.load(ref_flair_path).get_fdata()
ref_flair_slice = np.rot90(ref_flair_data[:,:,260])
print(ref_flair_data.shape)
data = {'mask': mask_slice, 'flair': fl_slice.copy(), 'ref_flair': ref_flair_slice}
data['flair'] = data['flair'] / np.max(data['flair'])
data = augmentations(data, 1, 128)
plt.figure()
plt.subplot(221)
plt.imshow(data['org_flair'], cmap='gray')
# plt.show()
plt.subplot(222)
plt.imshow(data['flair'], cmap='gray')
# plt.show()
plt.subplot(223)
plt.imshow(data['mask'], cmap='gray')
plt.subplot(224)
plt.imshow(data['flair'] - data['org_flair'], cmap='gray')
# plt.imshow(data['mask'], cmap='gray', alpha=0.5)
plt.show()