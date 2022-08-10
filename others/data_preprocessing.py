import os
import nibabel as nib
import numpy as np
from shutil import copyfile


def lit_dataset(org_folder=None, new_folder=None):
    org_folder = os.path.expanduser(org_folder)
    new_folder = os.path.expanduser(new_folder)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    subfolders = ((1,5), (6,10), (11, 15), (16,20), (21,25), (26,30))
    modality_to_org = {'t1': 'T1W', 't2': 'T2W', 'flair': 'FLAIR', 'ce': 'T1WKS'}
    for start, end in subfolders:
        for i in range(start, end + 1):
            prefix = 'patient%02d' % i
            new_prefix = 'patient_%02d' % i
            subject_folder = os.path.join(org_folder, 'patient%02d-%02d' % (start, end), prefix)
            brain_mask_path = os.path.join(subject_folder, prefix+'_brainmask.nii.gz')
            brain_mask = nib.load(brain_mask_path).get_fdata().astype(np.int64)
            for mod in modality_to_org.keys():
                mod_path = os.path.join(subject_folder, prefix + '_' + modality_to_org[mod] + '.nii.gz')
                img = nib.load(mod_path)
                new_img = nib.Nifti1Image(img.get_fdata().astype(np.int64) * brain_mask, img.affine, header=img.header)
                new_path = os.path.join(new_folder, new_prefix + '_01_' + mod + '.nii.gz')
                nib.save(new_img, new_path)
                print(mod_path, new_path)
            gt_path = os.path.join(subject_folder, prefix+'_consensus_gt.nii.gz')
            copyfile(gt_path, os.path.join(new_folder, new_prefix + '_01_mask.nii.gz'))


def covid_dataset(org_folder=None, new_folder=None):
    org_folder = os.path.expanduser(org_folder)
    new_folder = os.path.expanduser(new_folder)
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    nums = set()
    for ct_file in os.listdir(org_folder):
        if ct_file.endswith('_ct.nii.gz'):
            cur_num = int(ct_file.split('.')[0].split('-')[-1].split('_')[0])
            assert cur_num not in nums
            nums.add(cur_num)
            old_ct_path = os.path.join(org_folder, ct_file)
            new_ct_path = os.path.join(new_folder, 'covid_%d_01_ct.nii.gz' % cur_num)
            old_seg_path = old_ct_path.replace('_ct.nii.gz', '_seg.nii.gz')
            new_seg_path = os.path.join(new_folder, 'covid_%d_01_mask.nii.gz' % cur_num)
            copyfile(old_ct_path, new_ct_path)
            copyfile(old_seg_path, new_seg_path)


if __name__ == '__main__':
    lit_dataset('~/Documents/Datasets/LIT/', '~/Documents/Datasets/lit_dataset/')
    # covid_dataset('~/Documents/Datasets/COVID-19-20_v2/Train/', '~/Documents/Datasets/covid_dataset/')