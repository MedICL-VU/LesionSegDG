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


def registration(infile, outfile, fixed_image):
    # if check_if_exist(outfile):
    #     return

    if infile == fixed_image:
        copyfile(infile, outfile)
    else:
        reg = ants.Registration()
        reg.inputs.fixed_image = fixed_image
        reg.inputs.moving_image = infile
        reg.inputs.output_warped_image = outfile
        reg.inputs.metric = ['Mattes']
        # reg.inputs.metric_weight = [1]
        reg.inputs.shrink_factors = [[2,1]]
        reg.inputs.smoothing_sigmas = [[1,0]]
        reg.inputs.transforms = ['Affine']
        reg.inputs.transform_parameters = [(2.0,)]
        reg.inputs.number_of_iterations = [[1500, 200]]

        print("Time: " + time.asctime(time.localtime(time.time())))
        print("Running: " + reg.cmdline)
        res = reg.run()


# subjects = [1, 2, 5, 6, 10, 11, 13, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 39, 40, 42,
# 43, 44, 45, 46, 47, 48, 50, 51, 52, 55, 56, 63, 64, 65, 68, 69, 71, 72, 73, 76, 78, 79]
subjects = [1, 2]
dict_mod = {'flair': 'run-001_FLAIRstar',  't1': 'run-001_T1w', 't2': 'run-001_T2w'}


def register_and_move():
    base_org_dir = os.path.expanduser('~/Documents/Datasets/UVMdata1')
    base_new_dir = os.path.expanduser('~/Documents/Datasets/uvm1_dataset/reg')
    for s in subjects:
        subject_dir = os.path.join(base_org_dir, 'sub-%03d' % s, 'ses-001', 'anat')
        org_flair = os.path.join(subject_dir, 'sub-%03d_ses-001_run-001_FLAIRstar.nii.gz' % s)
        for mod in dict_mod.keys():
            org_mod = os.path.join(subject_dir, 'sub-%03d_ses-001_%s.nii.gz' % (s, dict_mod[mod]))
            new_mod = os.path.join(base_new_dir, 'uvm1_%02d_01_%s.nii.gz' % (s, mod))
            registration(org_mod, new_mod, org_flair)


def pre_process1():
    source_dir = os.path.expanduser('~/Documents/Datasets/uvm1_dataset/reg')
    target_dir = os.path.expanduser('~/Documents/Datasets/uvm1_dataset/bet')
    for s in subjects:
        source_file = os.path.join(source_dir, 'uvm1_%02d_01_flair.nii.gz'  % s)
        target_file = os.path.join(target_dir, 'uvm1_%02d_01_flair.nii.gz' % s)
        skull_stripping(source_file, target_file, True)

    for s in subjects:
        roi_file = os.path.join(target_dir, 'uvm1_%02d_01_flair_mask.nii.gz' % s)
        for mod in dict_mod.keys():
            if mod == 'flair':
                continue
            source_file = os.path.join(source_dir, 'uvm1_%02d_01_%s.nii.gz' % (s, mod))
            target_file = os.path.join(target_dir, 'uvm1_%02d_01_%s.nii.gz' % (s, mod))
            apply_mask(source_file, target_file, roi_file)


def bias_correction_parallel():
    source_folder = '/home/huahong/Documents/Datasets/uvm1_dataset/bet/'
    target_folder = '/home/huahong/Documents/Datasets/uvm1_dataset/raw/'
    for s in subjects:
        cmd = ""
        for i, mod in enumerate(dict_mod.keys()):
            source_path = os.path.join(source_folder, 'uvm1_%02d_01_%s.nii.gz' % (s, mod))
            target_path = os.path.join(target_folder, 'uvm1_%02d_01_%s.nii.gz' % (s, mod))
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