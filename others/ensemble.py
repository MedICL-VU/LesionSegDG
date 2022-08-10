import os
import nibabel as nib
import numpy as np
from test import seg_metrics, print_metrics
from collections import OrderedDict, defaultdict


results_name = [
  'colab_ms2008_3mod_aug3',
  'colab_ms2008_isbi_3mod_aug3_d200',
  'colab_ms2008_msseg_3mod_aug3_d200',
  'colab_ms2008_3mod_baseline',
  'colab_ms2008_isbi_3mod_baseline_d200',
  'colab_ms2008_msseg_3mod_baseline_d200'
]
base_dir = '/content/chb_dataset/'
suffix = 'nii.gz'
output_name = 'udg'
ret_metrics = defaultdict(list)
ret_mean = {}

# isbi, val_all
# i_range = range(1, 11)
# j_range = range(1, 2)
# mask_range = [1, 2]

# chb, raw
prefix = 'trainCHB'
folder = 'raw'
i_range = range(1, 11)
j_range = [1]
mask_range = [1]
for i in i_range:
    for j in j_range:
        for mask in mask_range:
            # mask_name = '%02d_%02d_mask%d_mask.%s' % (i, j, mask, suffix)
            # membership_name = '%02d_%02d_mask1_membership_%s.%s' % (i, j, res, suffix)
            # mask_pred_name = base_dir + 'val_all/%02d_%02d_mask1_pred_%s.%s' % (i, j, output_name, suffix)
            mask_name = '%s_%02d_%02d_mask.%s' % (prefix, i, j, suffix)
            mask_pred_name = base_dir + '%s/%s_%02d_%02d_pred_%s.%s' % (folder, prefix, i, j, output_name, suffix)
            if not os.path.exists(base_dir + '%s/%s' % (folder, mask_name)):
                continue
            mask_path = base_dir + '%s/%s' % (folder, mask_name)
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(np.int8)
            mask_pred = np.zeros_like(mask_data).astype(np.float)
            for res in results_name:
                membership_name = '%s_%02d_%02d_membership_%s.%s' % (prefix, i, j, res, suffix)
                cur_pred = nib.load(base_dir + '%s/%s' % (folder, membership_name)).get_fdata()
                mask_pred += cur_pred
            mask_pred /= len(results_name)
            mask_pred = (mask_pred > 0.5).astype(np.int8)
            res_this_mask = seg_metrics(mask_pred, mask_data, output_errors=False, fast_eval=False)
            metrics = list(res_this_mask.keys())
            for k in metrics:
                ret_metrics[k].append(res_this_mask[k])
            print_metrics('processed ' + '%s/%s' % (folder, mask_name) + '*,', res_this_mask)
            nib.Nifti1Image(mask_pred, mask_img.affine, mask_img.header).to_filename(mask_pred_name)
metrics = list(ret_metrics.keys())
for k in metrics:
    ret_mean[k] = np.mean(ret_metrics[k])
print_metrics('average: ', ret_mean)

ret_metrics = defaultdict(list)
ret_mean = {}

# # isbi, challenge
# for i in range(1, 15):
#     for j in range(1, 7):
#         if not os.path.exists(base_dir + 'challenge/test%02d_%02d_mask.nii' % (i, j)):
#             continue
#         mask_path = base_dir + 'challenge/test%02d_%02d_mask.nii' % (i, j)
#         mask_img = nib.load(mask_path)
#         mask_data = mask_img.get_fdata().astype(np.int8)
#         mask_pred = np.zeros_like(mask_data).astype(np.float)
#         for res in results_name:
#             cur_pred = nib.load(base_dir + 'challenge/test%02d_%02d_membership_%s.nii' % (i, j, res)).get_fdata()
#             mask_pred += cur_pred
#         mask_pred /= len(results_name)
#         mask_pred = (mask_pred > 0.5).astype(np.int8)
#         res_this_mask = seg_metrics(mask_pred, mask_data, output_errors=False, fast_eval=False)
#         metrics = list(res_this_mask.keys())
#         for k in metrics:
#             ret_metrics[k].append(res_this_mask[k])
#         print_metrics('processed ' + 'test%02d_%02d_mask.nii' % (i, j) + '*,', res_this_mask)
#         mask_pred_name = base_dir + 'challenge/test%02d_%02d_pred_%s.nii' % (i, j, output_name)
#         nib.Nifti1Image(mask_pred, mask_img.affine, mask_img.header).to_filename(mask_pred_name)
# metrics = list(ret_metrics.keys())
# for k in metrics:
#     ret_mean[k] = np.mean(ret_metrics[k])
# print_metrics('average: ', ret_mean)
