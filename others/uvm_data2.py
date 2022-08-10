import os
import zipfile
import nibabel as nib
import time
import subprocess
import shutil
import SimpleITK as sitk
from shutil import copyfile, move
from nipype.interfaces import fsl, ants
from others.miccai2008 import check_if_exist, skull_stripping, apply_mask, bias_correction
from others.uvm_data1 import registration


# subjects = [61, 80, 83, 88, 89, 91, 92, 93, 95, 97, 99, 101, 103, 106, 107, 108, 110, 111, 112, 114]
subjects = [83, 88,]
dict_mod = {'flair': 'FLAIR',  't1': 'T1w', 't2': 'part-mag_T2star'}


def register_and_move():
    base_org_dir = os.path.expanduser('~/Documents/Datasets/UVMdata2')
    base_new_dir = os.path.expanduser('~/Documents/Datasets/uvm2_dataset/reg')
    for s in subjects:
        subject_dir = os.path.join(base_org_dir, 'sub-%03d' % s, 'ses-01', 'anat')
        org_flair = os.path.join(subject_dir, 'sub-%03d_ses-01_run-001_FLAIR.nii.gz' % s)
        for mod in dict_mod.keys():
            org_mod = os.path.join(subject_dir, 'sub-%03d_ses-01_run-001_%s.nii.gz' % (s, dict_mod[mod]))
            new_mod = os.path.join(base_new_dir, 'uvm2_%02d_01_%s.nii.gz' % (s, mod))
            registration(org_mod, new_mod, org_flair)


def pre_process1():
    source_dir = os.path.expanduser('~/Documents/Datasets/uvm2_dataset/reg')
    target_dir = os.path.expanduser('~/Documents/Datasets/uvm2_dataset/bet')
    for s in subjects:
        source_file = os.path.join(source_dir, 'uvm2_%02d_01_flair.nii.gz'  % s)
        target_file = os.path.join(target_dir, 'uvm2_%02d_01_flair.nii.gz' % s)
        skull_stripping(source_file, target_file, True)

    for s in subjects:
        roi_file = os.path.join(target_dir, 'uvm2_%02d_01_flair_mask.nii.gz' % s)
        for mod in dict_mod.keys():
            if mod == 'flair':
                continue
            source_file = os.path.join(source_dir, 'uvm2_%02d_01_%s.nii.gz' % (s, mod))
            target_file = os.path.join(target_dir, 'uvm2_%02d_01_%s.nii.gz' % (s, mod))
            apply_mask(source_file, target_file, roi_file)


def bias_correction_parallel():
    source_folder = '/home/huahong/Documents/Datasets/uvm2_dataset/bet/'
    target_folder = '/home/huahong/Documents/Datasets/uvm2_dataset/raw/'
    for s in subjects:
        cmd = ""
        for i, mod in enumerate(dict_mod.keys()):
            source_path = os.path.join(source_folder, 'uvm2_%02d_01_%s.nii.gz' % (s, mod))
            target_path = os.path.join(target_folder, 'uvm2_%02d_01_%s.nii.gz' % (s, mod))
            n4 = ants.N4BiasFieldCorrection()
            n4.inputs.input_image = source_path
            n4.inputs.output_image = target_path
            if i > 0:
                cmd += " & "
            cmd += n4.cmdline
        print(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()


if __name__ == '__main__':
    register_and_move()
    pre_process1()
    bias_correction_parallel()