import os
import json
import shutil
import h5py
import numpy as np
import nibabel as nib
from util.util import mkdir
from util.image_property import hash_file, slice_with_neighborhood


def remove_folder_if_exist(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print('%s removed' % folder_path)
    else:
        print('%s does not exist' % folder_path)


class DataGenerator:
    def __init__(self, dataroot):
        self.sample_2d_count = 0
        self.sample_3d_count = 0
        self.dataroot = dataroot
        self.dir_this_phase = None
        self.axes = [0, 1, 2]
        self.rotation_2d = -1
        # self.all_2d_data = {'samples': [], 'raw_data': {}}
        with open(os.path.join(self.dataroot, 'properties.json'), 'r') as f:
            self.dic_properties = json.load(f)
        with open(os.path.join(self.dataroot, 'ids.json'), 'r') as f:
            self.dic_ids = json.load(f)

    def build_dataset(self, val_index, test_index, num_fold, dataset_phases, suffix):
        self.sample_2d_count, self.sample_3d_count = 0, 0
        self.samples = []
        val, test = 'val' in dataset_phases, 'test' in dataset_phases
        train = 'train' in dataset_phases

        all_ids = sorted(list(map(int, self.dic_ids.keys())))
        total_num = len(all_ids)
        print(all_ids)

        splits = [total_num // num_fold] * num_fold
        for i in range(total_num % num_fold):
            splits[i] += 1
        val_ids = [all_ids[i] for i in range(sum(splits[:val_index]), sum(splits[:val_index+1]))] if val else []
        test_ids = [all_ids[i] for i in range(sum(splits[:test_index]), sum(splits[:test_index+1]))] if test else []
        train_ids = [i for i in all_ids if i not in val_ids + test_ids]

        print(val_ids, test_ids)
        self.generate_general_data('val', val_ids, '3d', suffix=suffix)
        self.generate_general_data('test', test_ids, '3d', suffix=suffix)
        if train:
            self.generate_general_data('train', train_ids, '2d')

    def generate_general_data(self, phase, ids, mode, neighborhood=1, suffix="nii"):
        self.dir_this_phase = os.path.join(self.dataroot, phase)
        remove_folder_if_exist(self.dir_this_phase)
        mkdir(self.dir_this_phase)

        if mode == '2d':
            self.h5_data = h5py.File(os.path.join(self.dir_this_phase, 'data.h5'), 'a')
            self.grp_raw_data = self.h5_data.create_group("raw_data")
        for subject_id in ids:
            timepoints = self.dic_ids[str(subject_id)].keys()
            for timepoint in timepoints:
                masks = self.dic_ids[str(subject_id)][str(timepoint)]['mask'].keys()
                if mode == '3d':
                    self.sample_3d_count += 1
                for mask in masks:
                    if mode == '3d':
                        self.generate_3d_data(subject_id, timepoint, mask, phase, suffix)
                    else:
                        self.generate_2d_data(subject_id, timepoint, mask, neighborhood)
        if mode == '2d':
            self.samples = np.array(self.samples, dtype=h5py.special_dtype(vlen=str))
            self.h5_data.create_dataset('samples', data=self.samples)

    def generate_2d_data(self, subject_id, timepoint, mask, neighborhood):
        modalities = self.dic_ids[str(subject_id)][str(timepoint)]['modalities']
        path_mask = self.dic_ids[str(subject_id)][str(timepoint)]['mask'][mask]
        image_mask = nib.load(path_mask)
        voxel_sizes = nib.affines.voxel_sizes(image_mask.affine)
        image_data = {'mask': np.array(image_mask.get_fdata(), dtype=np.float32)}
        for modality in modalities:
            path_modality = self.dic_ids[str(subject_id)][str(timepoint)]['modalities'][modality]
            hash_label = hash_file(path_modality)
            modality_peak = self.dic_properties[hash_label]['peak']
            if modality == "flair":
                a = np.where(nib.load(path_modality).get_fdata().astype(np.int32) != 0)
            image_data[modality] = np.array(nib.load(path_modality).get_fdata() / modality_peak, dtype=np.float32)

        dataset_name = self.dataroot.split('/')[-1]
        # self.all_2d_data['raw_data'][(dataset_name, subject_id, timepoint, mask)] = image_data
        bbox = (np.min(a[0]), np.max(a[0])), (np.min(a[1]), np.max(a[1])), (np.min(a[2]), np.max(a[2]))
        sl = (slice(bbox[0][0], bbox[0][1], 1), slice(bbox[1][0], bbox[1][1], 1), slice(bbox[2][0], bbox[2][1], 1))
        for modality in list(modalities.keys()) + ['mask']:
            dataset_index = ';'.join(map(str, (dataset_name, subject_id, timepoint, mask, modality)))
            image_data[modality] = image_data[modality][sl]
            self.grp_raw_data.create_dataset(dataset_index, data=image_data[modality])

        data_to_save = {i: [] for i in modalities}
        data_to_save['mask'] = []
        for axis in self.axes:
            ratio = [k for i, k in enumerate(voxel_sizes) if i != axis]
            ratio = ratio[0] / ratio[1]  # if there is no rot90 following, it should be ratio[1]/ratio[0]
            slices_per_image = image_data['mask'].shape[axis]
            # print("Slices per image %d, current samples %d" % (slices_per_image, self.sample_2d_count))
            for i in range(slices_per_image):
                slice_mask = slice_with_neighborhood(image_data['mask'], axis, i, 0)
                if np.count_nonzero(slice_mask) < 2:
                    continue
                sample_data = tuple(map(str, (dataset_name, subject_id, timepoint, mask, axis, i, ratio)))
                self.samples.append(sample_data)
                self.sample_2d_count += 1

    def generate_3d_data(self, subject_id, timepoint, mask, phase, suffix):
        modalities = self.dic_ids[str(subject_id)][str(timepoint)]['modalities']
        for modality in modalities:
            path_src = self.dic_ids[str(subject_id)][str(timepoint)]['modalities'][modality]
            path_dst = os.path.join(self.dir_this_phase, mask + '_%s%d_%s.%s' % (phase, self.sample_3d_count, modality, suffix))
            shutil.copyfile(path_src, path_dst)
        path_src_mask = self.dic_ids[str(subject_id)][str(timepoint)]['mask'][mask]
        path_dst_mask = os.path.join(self.dir_this_phase, mask + '_%s%d_mask.%s' % (phase, self.sample_3d_count, suffix))
        shutil.copyfile(path_src_mask, path_dst_mask)

    def get_info(self):
        return {'sample_2d_count': self.sample_2d_count, 'sample_3d_count': self.sample_3d_count,
                'dataroot': self.dataroot}


if __name__ == '__main__':
    seed = 10
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    dataroot = os.path.join(os.path.expanduser("~"), "Documents", "Datasets", "isbi_dataset")
    data_generator = DataGenerator(dataroot)