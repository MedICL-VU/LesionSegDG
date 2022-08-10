import os
import nibabel as nib
import numpy as np
import time
import subprocess
import xml.etree.ElementTree as ET
from skimage import measure
from scipy.stats import pearsonr
from collections import OrderedDict, defaultdict


def seg_metrics(seg_vol, truth_vol, output_errors=False, fast_eval=False, cumulative=False, rm_empty=False):
    time_start = time.time()
    eps = 1e-6
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    if rm_empty and truth_total < 1:
        return OrderedDict([('dice', np.nan), ('ppv', np.nan), ('tpr', np.nan), ('lfpr', np.nan),
                            ('ltpr', np.nan), ('vd', np.nan), ('corr', np.nan), ('score', np.nan)])
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = 2 * tp / (seg_total + truth_total + eps)
    ppv = tp / (seg_total + eps)
    tpr = tp / (truth_total + eps)
    vd = abs(seg_total - truth_total) / (truth_total + eps)

    # calculate Pearson's correlation coefficient
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]

    reduce_param = tuple([i // 300 + 1 for i in seg_vol.shape])
    if fast_eval and max(reduce_param) > 1:
        seg_vol = (measure.block_reduce(seg_vol, (1, 2, 2), np.mean) > 0.5).astype(np.int8)
        truth_vol = (measure.block_reduce(truth_vol, (1, 2, 2), np.mean) > 0.5).astype(np.int8)

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = min(lfp_cnt / (seg_num + eps), 1.0)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / (truth_num + eps)

    score = dice / 8 + ppv / 8 + (1 - lfpr) / 4 + ltpr / 4 + corr / 4
    # print("Timed used calculating metrics: ", time.time() - time_start)

    if cumulative:
        # note corr is simply returned
        return OrderedDict([('seg_total', seg_total), ('truth_total', truth_total), ('tp', tp), ('corr', corr),
                            ('lfp_cnt', lfp_cnt), ('seg_num', seg_num), ('ltp_cnt', ltp_cnt), ('truth_num', truth_num)])
    return OrderedDict([('dice', dice), ('ppv', ppv), ('tpr', tpr), ('lfpr', lfpr),
                        ('ltpr', ltpr), ('vd', vd), ('corr', corr), ('score', score)])


def print_metrics(prefix, metrics):
    message = prefix + ' '
    for k, v in metrics.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


results_name = [
    # 'colab_msseg2021_twotp_baseline_L2',
    # 'colab_msseg2021_twotp_augs_L2',
    'colab_msseg2021_twotp_baseline_L2_IN',
    # 'colab_msseg2021_twotp_augs_L2_IN',
    # 'colab_msseg2021_tpwithmem_baseline_L2',
    # 'colab_msseg2021_tpwithmem_augs_L2',
    # 'colab_msseg2021_tpwithmem_baseline_L2_IN',
    # 'colab_msseg2021_tpwithmem_augs_L2_IN'
]
base_dir = '/content/msseg2021_dataset/'
suffix = 'nii.gz'
output_name = 'ensemble'
ret_metrics = defaultdict(list)
ret_mean = {}
rm_empty = True
use_anima = True
anima_path = '/content/Anima-Binaries-4.0.1/animaSegPerfAnalyzer'

prefix = 'mask'
folder = 'val'
i_range = range(1, 11)
j_range = [1]
mask_range = [1]
for i in i_range:
    for j in j_range:
        for mask in mask_range:
            mask_name = '%s_val%d_mask.%s' % (prefix, i, suffix)
            mask_pred_name = base_dir + '%s/%s_val%d_pred_%s.%s' % (folder, prefix, i, output_name, suffix)
            if not os.path.exists(base_dir + '%s/%s' % (folder, mask_name)):
                continue
            mask_path = base_dir + '%s/%s' % (folder, mask_name)
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(np.int8)
            mask_pred = np.zeros_like(mask_data).astype(np.float)
            for res in results_name:
                membership_name = '%s_val%d_membership_%s.%s' % (prefix, i, res, suffix)
                cur_pred = nib.load(base_dir + '%s/%s' % (folder, membership_name)).get_fdata()
                mask_pred += cur_pred
            mask_pred /= len(results_name)
            mask_pred = (mask_pred > 0.5).astype(np.int8)
            nib.Nifti1Image(mask_pred, mask_img.affine, mask_img.header).to_filename(mask_pred_name)

            res_this_mask = {}
            # res_this_mask = seg_metrics(mask_pred, mask_data, output_errors=False, fast_eval=False, rm_empty=rm_empty)
            # metrics = list(res_this_mask.keys())
            # for k in metrics:
            #     ret_metrics[k].append(res_this_mask[k])

            if use_anima:
                cmd = f"{anima_path} -o {mask_name} -r {mask_path} -i {mask_pred_name} -s -l -d -X"
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                process.wait()
                tree = ET.parse(f'{mask_name}_global.xml')
                root = tree.getroot()
                if np.isnan(np.float(root[-1].text)):
                    continue
                for elem in root:
                    res_this_mask[elem.attrib['name']] = np.float(elem.text)
                    ret_metrics[elem.attrib['name']].append(np.float(elem.text))

            print_metrics('processed ' + '%s/%s' % (folder, mask_name) + '*,', res_this_mask)

metrics = list(ret_metrics.keys())
for k in metrics:
    # if rm_empty:
    #     ret_metrics[k] = np.array(ret_metrics[k])
    #     ret_metrics[k] = ret_metrics[k][np.isfinite(ret_metrics[k])]
    ret_mean[k] = np.mean(ret_metrics[k])
print_metrics('average: ', ret_mean)


