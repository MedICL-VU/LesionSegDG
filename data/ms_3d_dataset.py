import os.path
import numpy as np
import json
import nibabel as nib
from data.base_dataset import BaseDataset
from util.image_property import hash_file, normalize_image, slice_with_neighborhood


def get_3d_paths(dir, config):
    images = []
    MODALITIES = config["modalities"]
    SUFFIX = config["suffix"]
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(MODALITIES[0]+'.' + SUFFIX):
                images.append([])
                images[-1].append(os.path.join(root, fname))
                for i in range(1, len(MODALITIES)):
                    images[-1].append(os.path.join(root, fname.replace(MODALITIES[0]+'.'+SUFFIX, MODALITIES[i]+'.' + SUFFIX)))
                images[-1].append(os.path.join(root, fname.replace(MODALITIES[0] + '.' + SUFFIX, 'mask.' + SUFFIX)))

    images.sort(key=lambda x: x[0])
    return images


def flip_by_times(np_array, times):
    for i in range(times):
        np_array = np.flip(np_array, axis=1)
    return np_array


# currently, this dataset is for test use only
class Ms3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        with open('config.json', 'r') as f:
            config = json.load(f)
        self.all_modalities = opt.modalities.split(',')
        for dataset in config:
            if os.path.expanduser(config[dataset]['path']) == os.path.expanduser(opt.dataroot):
                self.config = config[dataset]
                self.modalities = self.config["modalities"]
                self.suffix = self.config["suffix"]
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.all_paths = get_3d_paths(self.dir_AB, self.config)
        self.neighbors = opt.input_nc // 2
        if os.path.exists(os.path.join(opt.dataroot, 'properties.json')):
            with open(os.path.join(opt.dataroot, 'properties.json'), 'r') as f:
                self.dic_properties = json.load(f)
        else:
            self.dic_properties = {}

    def __getitem__(self, index):
        AXIS_TO_TAKE = self.config["axis_to_take"]
        paths_this_scan = self.all_paths[index]
        voxel_sizes = nib.affines.voxel_sizes(nib.load(paths_this_scan[0]).affine)
        data_all_modalities = {}
        for i, modality in enumerate(self.modalities):
            path_modality = paths_this_scan[i]
            label_modality = hash_file(path_modality)
            data_modality = nib.load(path_modality).get_fdata()
            if label_modality in self.dic_properties:
                peak_modality = self.dic_properties[label_modality]['peak']
            else:
                peak_modality = normalize_image(data_modality, modality)
            data_all_modalities[modality] = np.array(data_modality / peak_modality, dtype=np.float32)

        if os.path.exists(paths_this_scan[-1]):
            data_all_modalities['mask'] = nib.load(paths_this_scan[-1]).get_fdata()

        for modality in self.all_modalities:
            if modality not in self.modalities:
                data_all_modalities[modality] = np.zeros_like(data_all_modalities[self.modalities[0]])

        data_return = {mod: None for mod in self.all_modalities + ['mask']}
        data_return['org_size'] = {'axial': None, 'sagittal': None, 'coronal': None}
        data_return['ratios'] = {'axial': None, 'sagittal': None, 'coronal': None}
        data_return['mask_paths'] = paths_this_scan[-1]
        data_return['alt_paths'] = paths_this_scan[0]

        a = np.where(data_all_modalities['flair'] != 0)
        bbox = (np.min(a[0]), np.max(a[0])), (np.min(a[1]), np.max(a[1])), (np.min(a[2]), np.max(a[2]))
        data_return['bbox'] = bbox
        sl = (slice(bbox[0][0], bbox[0][1], 1), slice(bbox[1][0], bbox[1][1], 1), slice(bbox[2][0], bbox[2][1], 1))

        for k, orientation in enumerate(['axial', 'sagittal', 'coronal']):
            ratio = [size for axis, size in enumerate(voxel_sizes) if axis != AXIS_TO_TAKE[k]]
            ratio = ratio[1] / ratio[0]
            data_return['ratios'][orientation] = ratio
            cur_shape = data_all_modalities[self.modalities[0]].shape
            data_return['org_size'][orientation] = \
                tuple([axis_len for axis, axis_len in enumerate(cur_shape) if axis != AXIS_TO_TAKE[k]])
        data_return['org_size']['all'] = data_all_modalities[self.modalities[0]].shape
            
        for modality in self.all_modalities:
            data_return[modality] = data_all_modalities[modality][sl] / 2 -1
        if os.path.exists(paths_this_scan[-1]):
            data_return['mask'] = data_all_modalities['mask'] * 2 - 1
        else:
            data_return['mask'] = np.zeros_like(data_all_modalities[self.modalities[0]])

        return data_return

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'Ms3dDataset'
