import os
import subprocess
import shutil
import nibabel as nib
import numpy as np
from nipype.interfaces import fsl, ants
from skimage import measure
from others.miccai2008 import check_if_exist, skull_stripping, bias_correction
from collections import defaultdict

subjects = ['013', '015', '016', '018', '019', '020', '021', '024', '026', '027', '029', '030', '032', '035',
            '037', '039', '043', '047', '048', '049', '051', '052', '057', '061', '068', '069', '070', '074',
            '077', '083', '084', '088', '089', '090', '091', '094', '095', '096', '099', '100']


def pre_process1():
    source_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/unprocessed/'
    target_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/BET/'
    for subject in subjects:
        for timepoint in [1, 2]:
            source_file_name = f'flair_time0{timepoint}_on_middle_space.nii.gz'
            target_file_name = f'training_{subject}_01_flair{timepoint}.nii.gz'
            source_path = os.path.join(source_folder, subject, source_file_name)
            target_path = os.path.join(target_folder, target_file_name)
            skull_stripping(source_path, target_path, True)


def pre_process2():
    source_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/unprocessed/'
    target_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/raw/'
    for subject in subjects:
        source_file_name = 'ground_truth.nii.gz'
        target_file_name = f'training_{subject}_01_mask_new.nii.gz'
        source_path = os.path.join(source_folder, subject, source_file_name)
        target_path = os.path.join(target_folder, target_file_name)
        shutil.copyfile(source_path, target_path)


def pre_process3_parallel():
    source_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/BET/'
    target_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/raw/'
    for subject in subjects:
        cmd = ""
        for timepoint in [1, 2]:
            source_path = os.path.join(source_folder, f'training_{subject}_01_flair{timepoint}.nii.gz')
            target_path = os.path.join(target_folder, f'training_{subject}_01_flair{timepoint}.nii.gz')
            n4 = ants.N4BiasFieldCorrection()
            n4.inputs.input_image = source_path
            n4.inputs.output_image = target_path
            if timepoint > 1:
                cmd += " & "
            cmd += n4.cmdline
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()


def dataset_properties():
    target_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/raw_custom/'
    seg_num_total, sum_total = defaultdict(int), defaultdict(int)
    for i, subject in enumerate(subjects):
        mask_path = os.path.join(target_folder, f'training_{subject}_01_mask_new.nii.gz')
        mask_data = nib.load(mask_path).get_fdata()
        seg_labels, seg_num = measure.label(mask_data, return_num=True, connectivity=2)
        seg_num_total[i // 8] += seg_num
        sum_total[i // 8] += int(np.sum(mask_data))
        min_voxel = min([np.sum(seg_labels==i) for i in range(1, seg_num+1)] + [1000000])
        print(i//8, subject, mask_data.shape, seg_num, int(np.sum(mask_data)), int(min_voxel))
    print(seg_num_total, sum_total)


def move_files_anima():
    source_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/processed_anima/'
    target_folder = '/home/huahong/Documents/Datasets/msseg2021_dataset/raw_anima/'
    for subject in subjects:
        for timepoint in [1, 2]:
            source_file_name = f'flair_time0{timepoint}_on_middle_space.nii.gz'
            target_file_name = f'training_{subject}_01_flair{timepoint}.nii.gz'
            source_path = os.path.join(source_folder, subject, source_file_name)
            target_path = os.path.join(target_folder, target_file_name)
            shutil.copyfile(source_path, target_path)
    for subject in subjects:
        source_file_name = 'ground_truth.nii.gz'
        target_file_name = f'training_{subject}_01_mask_new.nii.gz'
        source_path = os.path.join(source_folder, subject, source_file_name)
        target_path = os.path.join(target_folder, target_file_name)
        shutil.copyfile(source_path, target_path)


if __name__ == '__main__':
    dataset_properties()