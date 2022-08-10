import os.path
import random
import torchvision.transforms as transforms
import h5py
import json
from imgaug import augmenters as iaa
from data.base_dataset import BaseDataset
from util.image_property import slice_with_neighborhood
from util.from_albumentations import *


def get_2d_paths(dir):
    arrays = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.pkl'):
                path = os.path.join(root, fname)
                arrays.append(path)

    return arrays


def augmentations(data, ratio, opt, modalities_w_mask=None):
    height, width = data[modalities_w_mask[-1]].shape[:2]  # height/y for first axis, width/x for second axis
    nums = opt.augmentations.split(',')
    modalities = modalities_w_mask.copy()
    if 'mask' in modalities:
        modalities.remove('mask')

    if height < opt.trainSize or width < opt.trainSize:
        new_height = max(opt.trainSize, height)
        new_width = max(opt.trainSize, width)
        pad_params = (((new_height-height) // 2, new_height - height - (new_height-height) // 2),
                      ((new_width-width) // 2, new_width - width - (new_width-width) // 2), (0, 0))
        for modality in modalities:
            data[modality] = np.pad(data[modality], pad_params, 'constant', constant_values=0)
        if 'mask' in data:
            data['mask'] = np.pad(data['mask'], pad_params[:2], 'constant', constant_values=0)
        height, width = new_height, new_width

    # for ratio around 1 and p=0.2, random assign a big ratio
    if '1' in nums and abs(1 - ratio) < 0.05 and random.random() < 0.2:
        ratio = 1 + random.random()
        if random.random() < 0.5:
            ratio = 1 / ratio

    # p=0.8, reshape the image based on ratio
    if '2' in nums and abs(1 - ratio) > 0.05 and random.random() < 0.8:
        if ratio > 1:
            height, width = height, int(width * ratio)
        else:
            height, width = int(height / ratio), width

        for modality in modalities_w_mask:
            interpolation = cv2.INTER_LINEAR
            data[modality] = cv2.resize(data[modality], (width, height), interpolation)

    # p=0.3, randomly rotate angle between 1->45
    if '3' in nums and random.random() < 0.3:
        angle = random.randint(1, 45)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        for modality in modalities_w_mask:
            interpolation = cv2.INTER_LINEAR
            data[modality] = cv2.warpAffine(data[modality], M=matrix, dsize=(width, height), flags=interpolation)

    # p=0.3, randomly distort image
    if '4' in nums and random.random() < 0.3:
        num_steps, distort_limit = 5, (-0.3, 0.3)
        xsteps = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in range(num_steps + 1)]
        ysteps = [1 + random.uniform(distort_limit[0], distort_limit[1]) for i in range(num_steps + 1)]
        for modality in modalities_w_mask:
            data[modality] = grid_distortion(data[modality], num_steps, xsteps, ysteps,
                                             border_mode=cv2.BORDER_CONSTANT, value=0)

    # p=0.3, do random crop and resize to trainSize
    need_resize = False
    if '5' in nums and random.random() < 0.3:
        crop_size = random.randint(int(opt.trainSize / 1.5), min(height, width))
        need_resize = True
    else:
        crop_size = opt.trainSize

    mask = data['mask'] if 'mask' in data else data[modalities_w_mask[-1]][:, :, 1]
    if np.sum(mask) < 1e-3 or random.random() < 0.05:
        x_min = random.randint(0, width - crop_size)
        y_min = random.randint(0, height - crop_size)
    else:
        non_zero_yx = np.argwhere(mask)
        y, x = random.choice(non_zero_yx)
        x_min = x - random.randint(0, crop_size - 1)
        y_min = y - random.randint(0, crop_size - 1)
        x_min = np.clip(x_min, 0, width - crop_size)
        y_min = np.clip(y_min, 0, height - crop_size)

    for modality in modalities_w_mask:
        interpolation = cv2.INTER_LINEAR
        data[modality] = data[modality][y_min:y_min+crop_size, x_min:x_min+crop_size]
        if need_resize:
            data[modality] = cv2.resize(data[modality], (opt.trainSize, opt.trainSize), interpolation)
    height, width = opt.trainSize, opt.trainSize

    # p=0.3, randomly flip
    for axis in [1, 0]:
        if random.random() < 0.3:
            for modality in modalities_w_mask:
                data[modality] = np.flip(data[modality], axis).copy()

    # p=0.5, randomly transpose
    if random.random() < 0.5:
        height, width = width, height
        for modality in modalities_w_mask:
            transpose_index = (1, 0, 2) if modality != 'mask' else (1, 0)
            data[modality] = np.transpose(data[modality], transpose_index)

    # randomly blur each modal independently
    for modality in modalities:
        if '8' in nums and random.random() < 0.2:
            ksize = int(random.choice([3, 3, 3, 5, 5, 7]))
            data[modality] = blur(data[modality], ksize)

    # randomly downscale each modal independently
    for modality in modalities:
        if '9' in nums and random.random() < 0.1:
            data[modality] = downscale(data[modality], scale=random.uniform(0.5, 1.0))

    # randomly sharpen
    for modality in modalities:
        if '12' in nums and random.random() < 0.2:
            data[modality] = iaa.Sharpen(alpha=(0, 0.5), lightness=(0.5, 1.5)).augment_image(data[modality])

    # randomly Gaussian noise
    for modality in modalities:
        if '13' in nums and random.random() < 0.2:
            data[modality] += np.random.normal(random.uniform(0, 0.1), random.uniform(0, 0.2), data[modality].shape)

    for modality in modalities:
        if '16' in nums and random.random() < 0.2:
            data[modality] = np.ones_like(data[modality]) * random.random()

    for modality in ['pd', 'ce']:
        if '17' in nums and modality in modalities and random.random() < 0.3:
            data[modality] = np.ones_like(data[modality]) * random.random()

    if 'mask' in data:
        data['mask'] = (data['mask'] > 0.5).astype(np.float32)
    return data


def histogram_augmentation(data, opt, modalities=None):
    nums = opt.augmentations.split(',')
    for modality in modalities:
        if '10' in nums and random.random() < 0.2:
            limit = random.uniform(4, 1)
            max_value, min_value = np.max(data[modality]), np.min(data[modality])
            if max_value - min_value > 1e-6:
                data_uint8 = ((data[modality] - min_value) / (max_value - min_value) * 255).astype(np.uint8)
                data[modality] = clahe(data_uint8, limit, (8, 8)).astype(np.float32) / 255 * (max_value - min_value) + min_value

        if '11' in nums and random.random() < 0.2:
            brightness = random.uniform(0.65, 1.35)
            contrast = random.uniform(0.65, 1.35)
            transforms = [
                lambda x: adjust_brightness_torchvision(x, brightness),
                lambda x: adjust_contrast_torchvision(x, contrast),
            ]
            random.shuffle(transforms)

            max_value, min_value = np.max(data[modality]), np.min(data[modality])
            if max_value - min_value > 1e-6:
                data_uint8 = ((data[modality] - min_value) / (max_value - min_value) * 255).astype(np.uint8)
                for transform in transforms:
                    data_uint8 = transform(data_uint8)
                data[modality] = data_uint8.astype(np.float32) / 255 * (max_value - min_value) + min_value

        # radial gradient
        if '14' in nums and random.random() < 0.2:
            h, w = data[modality].shape[:2]
            circle_w = random.randint(10, w)
            circle_h = min(random.randint(circle_w // 2, circle_w * 2), h)
            Y = np.linspace(-1, 1, circle_h)[:, None]
            X = np.linspace(-1, 1, circle_w)[None, :]
            gradient = X ** 2 + Y ** 2
            gradient = np.clip(a=gradient, a_min=0.0, a_max=1.0)
            gradient = 1 - gradient

            start_h = random.randint(0, h - circle_h)
            start_w = random.randint(0, w - circle_w)
            amp = random.uniform(-0.3, 0.3)
            multiplier = np.ones((h, w, 1))
            multiplier[start_h:start_h + circle_h, start_w:start_w + circle_w, 0] += gradient * amp
            data[modality] *= np.tile(multiplier, (1, 1, 3))

        # linear gradient horizontal or vertical
        if '15' in nums and random.random() < 0.2:
            h, w = data[modality].shape[:2]
            multiplier = np.ones((h, w, 1))
            amp = random.uniform(-0.3, 0.3)
            if random.random() < 0.5:
                rec_h = random.randint(10, h)
                start_h = random.randint(0, h - rec_h)
                gradient = np.tile(1 - np.linspace(-1, 1, rec_h) ** 2, (w, 1)).T
                multiplier[start_h:start_h + rec_h, :, 0] += gradient * amp
            else:
                rec_w = random.randint(10, w)
                start_w = random.randint(0, h - rec_w)
                gradient = np.tile(1 - np.linspace(-1, 1, rec_w) ** 2, (h, 1))
                multiplier[:, start_w:start_w + rec_w, 0] += gradient * amp
            data[modality] *= np.tile(multiplier, (1, 1, 3))

    return data


class MsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.all_labeled_samples = []
        self.raw_data = {}
        self.modalities_to_use = opt.modalities.split(',')
        with open('config.json', 'r') as f:
            config = json.load(f)
        for i in config:
            if not config[i]['train']:
                continue
            path_data = os.path.join(os.path.expanduser(config[str(i)]["path"]), opt.phase, 'data.h5')
            cur_data = h5py.File(path_data, 'r')
            assert config[i]['domain'] in ['source', 'target']
            if config[i]['domain'] != 'source':
                continue
            self.all_labeled_samples.extend(cur_data['samples'])
            use_mask = ['mask']
            visited = set()
            for dataset_name, subject_id, timepoint, mask, axis, ind, ratio in cur_data['samples']:
                for modality in config[i]['modalities'] + use_mask:
                    dataset_index = ';'.join((dataset_name.decode("utf-8") , subject_id.decode("utf-8") ,
                                              timepoint.decode("utf-8") , mask.decode("utf-8") , modality))
                    if dataset_index in visited:
                        continue
                    self.raw_data[(dataset_name, subject_id, timepoint, mask, modality)] = cur_data.get('raw_data/'+dataset_index)[:]
                    visited.add(dataset_index)
        random.shuffle(self.all_labeled_samples)
        random.shuffle(self.all_labeled_samples)
        self.len_labeled = len(self.all_labeled_samples)

    def __getitem__(self, index):
        data_return1 = self.__getitem__imp__(index)
        nums = self.opt.augmentations.split(',')
        if 'mixup' in nums:
            p = random.random() if random.random() < 0.3 else 1.0
            random_index = random.randint(0, self.len_labeled - 1)
            data_return2 = self.__getitem__imp__(random_index)
            for modality in self.modalities_to_use:
                data_return1[modality] = data_return1[modality] * p + data_return2[modality] * (1-p)
            data_return1['mask2'] = data_return2['mask']
            data_return1['weights'] = np.float32(p)
        return data_return1

    def __getitem__imp__(self, index):
        (l_dataset, l_subject_id, l_timepoint, l_mask, l_axis, l_index, l_ratio) = self.all_labeled_samples[index]
        data_labeled = {}
        available_modalities = []
        for modality in self.modalities_to_use + ['mask']:
            if (l_dataset, l_subject_id, l_timepoint, l_mask, modality) in self.raw_data:
                data_3d = self.raw_data[(l_dataset, l_subject_id, l_timepoint, l_mask, modality)]
                neighborhood = self.opt.input_nc // 2 if modality != 'mask' else 0
                data_labeled[modality] = slice_with_neighborhood(data_3d, int(l_axis), int(l_index), neighborhood)
                available_modalities.append(modality)

        available_modalities.remove('mask')
        data_labeled['mask'] = data_labeled['mask'][:, :, 0]
        data_labeled = augmentations(data_labeled, float(l_ratio), self.opt,
                                     modalities_w_mask=available_modalities + ['mask'])
        data_labeled = histogram_augmentation(data_labeled, self.opt, modalities=available_modalities)

        data_return = {**data_labeled}

        for modality in self.modalities_to_use:
            if modality not in available_modalities:
                data_return[modality] = np.zeros((self.opt.trainSize, self.opt.trainSize, self.opt.input_nc)).astype(
                    np.float32)
            data_return[modality] = data_return[modality] / 2 - 1
        data_return['mask'] = np.expand_dims(data_return['mask'], axis=2) * 2 - 1

        for modality in self.modalities_to_use + ['mask']:
            data_return[modality] = transforms.ToTensor()(data_return[modality])

        return data_return

    def __len__(self):
        return self.len_labeled

    def name(self):
        return 'MsDataset'