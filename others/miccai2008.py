import os
import zipfile
import nibabel as nib
import time
import subprocess
import shutil
import SimpleITK as sitk
from nipype.interfaces import fsl, ants

# 1-10 subjects available for both datasets
train_files = {'CHB': list(range(1, 11)), 'UNC': list(range(1, 11))}
# challenge, CHB: 1-15 is needed for submission, UNC: 1-10 is needed for submission
test_files = {'CHB': list(range(1, 19)), 'UNC': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}
all_files = {'train': train_files, 'test': test_files}


def check_if_exist(outfile):
    if os.path.exists(outfile):
        return True
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    return False


def skull_stripping(infile, outfile, ROI_file=None):
    # outfile = infile.replace('.nii.gz', '_bet.nii.gz')
    if check_if_exist(outfile):
        return

    skullstrip = fsl.BET()
    skullstrip.inputs.in_file = infile
    skullstrip.inputs.out_file = outfile
    # skullstrip.inputs.reduce_bias = True
    if ROI_file:
        skullstrip.inputs.mask = True
    print("Time: " + time.asctime(time.localtime(time.time())))
    print("Running: " + skullstrip.cmdline)
    res = skullstrip.run()
    print(res.outputs.out_file)


def apply_mask(infile, outfile, mask_file):
    if check_if_exist(outfile):
        return
    app = fsl.ApplyMask()
    app.inputs.in_file = infile
    app.inputs.mask_file = mask_file
    app.inputs.out_file = outfile
    print("Time: " + time.asctime(time.localtime(time.time())))
    print("Running: " + app.cmdline)
    res = app.run()


def bias_correction(infile, outfile):
    # outfile = infile.replace('.nii.gz', '_n4.nii.gz')
    if check_if_exist(outfile):
        return

    n4 = ants.N4BiasFieldCorrection()
    n4.inputs.input_image = infile
    n4.inputs.output_image = outfile
    print("Time: " + time.asctime(time.localtime(time.time())))
    print("Running: " + n4.cmdline)
    res = n4.run()


def unzip_files():
    source_folder = '/home/huahong/Documents/Datasets/MASI_2008/'
    train_file_list_CHB = ['CHB_train_Part1.zip', 'CHB_train_Part2.zip']
    train_file_list_UNC = ['UNC_train_Part1.zip', 'UNC_train_Part2.zip']
    test_file_list_CHB = ['CHB_test1_Part1.zip', 'CHB_test1_Part2.zip', 'CHB_test1_Part3.zip']
    test_file_list_UNC = ['UNC_test1_Part1.zip', 'UNC_test1_Part2.zip']

    for zip_file in train_file_list_CHB+train_file_list_UNC+test_file_list_CHB+test_file_list_UNC:
        print("unzipping ", zip_file)
        with zipfile.ZipFile(source_folder+zip_file, "r") as zip_ref:
            zip_ref.extractall(source_folder)


def move_files():
    source_folder = '/home/huahong/Documents/Datasets/MASI_2008/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/unprocessed/'

    for dataset in train_files:
        for subject in train_files[dataset]:
            subject_name = dataset + '_train_Case%02d' % subject
            print('train ', subject_name)
            for mod in ['lesion', 'lesion_byCHB']:
                source_path = os.path.join(source_folder, subject_name, subject_name + "_%s.nhdr" % mod)
                if not os.path.exists(source_path):
                    continue
                if dataset == 'CHB' and mod == 'lesion':
                    mod = 'mask'
                elif dataset == 'UNC' and mod == 'lesion':
                    mod = 'mask1'
                elif dataset == 'UNC' and mod == 'lesion_byCHB':
                    mod = 'mask2'
                target_path = os.path.join(target_folder, 'train%s_%02d_01_%s.nii.gz' % (dataset, subject, mod))
                img = sitk.ReadImage(source_path)
                sitk.WriteImage(img, target_path)

    # for phase in ['train', 'test']:
    #     for dataset in test_files:
    #         for subject in test_files[dataset]:
    #             if phase == 'train':
    #                 subject_name = dataset + '_train_Case%02d' % subject
    #             else:
    #                 subject_name = dataset + '_test1_Case%02d' % subject
    #             print(phase, ' ', subject_name)
    #             for mod in ['T1', 'T2', 'FLAIR']:
    #                 source_path = os.path.join(source_folder, subject_name, subject_name + "_%s.nhdr" % mod)
    #                 target_path = os.path.join(target_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod))
    #                 img = sitk.ReadImage(source_path)
    #                 sitk.WriteImage(img, target_path)


def pre_process1():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/unprocessed/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/BET/'
    for phase in ['train', 'test']:
        for dataset in all_files[phase]:
            for subject in all_files[phase][dataset]:
                T1_file_name = '%s%s_%02d_01_T1.nii.gz' % (phase,dataset, subject)
                source_path = os.path.join(source_folder, T1_file_name)
                target_path = os.path.join(target_folder, T1_file_name)
                skull_stripping(source_path, target_path, True)


def pre_process2():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/unprocessed/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/BET/'
    for phase in ['train', 'test']:
        for dataset in all_files[phase]:
            for subject in all_files[phase][dataset]:
                T1_ROI_name = '%s%s_%02d_01_T1_mask.nii.gz' % (phase, dataset, subject)
                ROI_path = os.path.join(target_folder, T1_ROI_name)
                for mod in ['T2', 'FLAIR']:
                    source_path = os.path.join(source_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod))
                    target_path = os.path.join(target_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod))
                    apply_mask(source_path, target_path, ROI_path)


def pre_process3():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/BET/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/raw/'
    for phase in ['train', 'test']:
        for dataset in all_files[phase]:
            for subject in all_files[phase][dataset]:
                for i, mod in enumerate(['T1', 'T2', 'FLAIR']):
                    source_path = os.path.join(source_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod))
                    target_path = os.path.join(target_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod.lower()))
                    bias_correction(source_path, target_path)


def pre_process3_parallel():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/BET/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/raw/'
    for phase in ['train', 'test']:
        for dataset in all_files[phase]:
            for subject in all_files[phase][dataset]:
                cmd = ""
                for i, mod in enumerate(['T1', 'T2', 'FLAIR']):
                    source_path = os.path.join(source_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod))
                    target_path = os.path.join(target_folder, '%s%s_%02d_01_%s.nii.gz' % (phase, dataset, subject, mod.lower()))
                    n4 = ants.N4BiasFieldCorrection()
                    n4.inputs.input_image = source_path
                    n4.inputs.output_image = target_path
                    if i > 0:
                        cmd += " & "
                    cmd += n4.cmdline
                print(cmd)
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()


def pre_process4():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/unprocessed/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/raw/'
    for dataset in train_files:
        for subject in train_files[dataset]:
            masks = ['mask1', 'mask2'] if dataset == 'UNC' else ['mask']
            for mask in masks:
                source_path = os.path.join(source_folder, 'train%s_%02d_01_%s.nii.gz' % (dataset, subject, mask))
                target_path = os.path.join(target_folder, 'train%s_%02d_01_%s.nii.gz' % (dataset, subject, mask))
                shutil.move(source_path, target_path)


def nii_to_nrrd():
    source_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/challenge/'
    target_folder = '/home/huahong/Documents/Datasets/miccai2008_dataset/Vandy/'
    pred_name = 'pred_vandy'
    for dataset in test_files:
        for subject in test_files[dataset]:
            source_path = os.path.join(source_folder, 'test%s_%02d_01_%s.nii.gz' % (dataset, subject, pred_name))
            target_path = os.path.join(target_folder, dataset + '_test1_Case%02d_segmentation.nrrd' % subject)
            img = sitk.ReadImage(source_path)
            sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), target_path)
            print(target_path)


if __name__ == '__main__':
    # unzip_files()
    # move_files()
    # pre_process1()
    # pre_process2()
    # pre_process3()
    # pre_process3_parallel()
    # pre_process4()
    nii_to_nrrd()