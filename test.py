import torch
import time
import cv2
import os
import json
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import subprocess
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
# from configurations import *
from util.image_property import hash_file, slice_with_neighborhood


def pad_images(opt_test, *image_list):
    padNum = -1
    pad_pos_y = (opt_test.testSize - image_list[0][0].shape[-2]) // 2
    pad_pos_x = (opt_test.testSize - image_list[0][0].shape[-1]) // 2
    pad_param = [pad_pos_x, opt_test.testSize - image_list[0].shape[-1] - pad_pos_x,
                 pad_pos_y, opt_test.testSize - image_list[0].shape[-2] - pad_pos_y]

    var_return = []
    image_list = list(image_list)
    for one_image in image_list:
        pad_image = F.pad(one_image, pad_param, 'constant', padNum)
        var_return += [pad_image]

    sl = [slice(None)] * 2
    sl[0] = slice(pad_pos_y, pad_pos_y + image_list[0][0].shape[-2], 1)
    sl[1] = slice(pad_pos_x, pad_pos_x + image_list[0][0].shape[-1], 1)
    var_return += [tuple(sl)]
    return var_return


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


def seg_metrics_cumulative(metrics):
    if len(metrics['seg_total']) == 0:
        return OrderedDict([('dice', 0), ('ppv', 0), ('tpr', 0), ('lfpr', 0),
                        ('ltpr', 0), ('vd', 0), ('corr', 0), ('score', 0)])

    eps = 1e-6
    seg_total = np.sum(metrics['seg_total'])
    truth_total = np.sum(metrics['truth_total'])
    tp = np.sum(metrics['tp'])
    dice = 2 * tp / (seg_total + truth_total + eps)
    ppv = tp / (seg_total + eps)
    tpr = tp / (truth_total + eps)
    vd = np.sum(abs(np.array(metrics['seg_total']) - np.array(metrics['truth_total']))) / truth_total
    corr_all = np.array(metrics['corr'])
    corr = np.mean(corr_all[np.isfinite(corr_all)])
    lfp_cnt = np.sum(metrics['lfp_cnt'])
    seg_num = np.sum(metrics['seg_num'])
    ltp_cnt = np.sum(metrics['ltp_cnt'])
    truth_num = np.sum(metrics['truth_num'])
    lfpr = min(lfp_cnt / (seg_num + eps), 1.0)
    ltpr = ltp_cnt / (truth_num + eps)
    score = dice / 8 + ppv / 8 + (1 - lfpr) / 4 + ltpr / 4 + corr / 4

    return OrderedDict([('dice', dice), ('ppv', ppv), ('tpr', tpr), ('lfpr', lfpr),
                        ('ltpr', ltpr), ('vd', vd), ('corr', corr), ('score', score)])


def print_metrics(prefix, metrics):
    message = prefix + ' '
    for k, v in metrics.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def model_test(models, dataset_test, opt_test, num_test, save_images=False, models_weight=None,
               mask_suffix='pred', save_membership=False):
    if not num_test:
        print("no %s subjects" % opt_test.phase)
    assert len(models), "no models loaded"

    start_time = time.time()
    orientations = ['axial', 'sagittal', 'coronal']
    transpose = {2: (1, 2, 0), 0: (0, 1, 2), 1: (1, 0, 2)}
    orientation_weight = [1, 1, 1]
    ret_metrics = defaultdict(list)
    ret_mean, ret_std = {'empty_lesion_number': 0, 'empty_lesion_volume': 0}, {}
    metrics = []
    fast_eval = opt_test.fast_eval

    with open('config.json', 'r') as f:
        config = json.load(f)
    this_config = {}
    for dataset in config:
        if os.path.expanduser(config[dataset]['path']) == opt_test.dataroot:
            this_config = config[dataset]
    MODALITIES = opt_test.modalities.split(',')
    SUFFIX = this_config["suffix"]
    AXIS_TO_TAKE = this_config["axis_to_take"]
    alt_modality = this_config['modalities'][0]

    dict_results = {}
    for i, data in enumerate(dataset_test):
        if i >= num_test:
            break

        mask, mask_path, alt_path = data['mask'], data['mask_paths'][0], data['alt_paths'][0]
        basename = os.path.basename(data['alt_paths'][0])
        basename = basename[:len(basename) - len(MODALITIES[0]) - len(SUFFIX) - 1]

        hash_label = hash_file(alt_path)
        if hash_label not in dict_results:
            mask_pred = 0
            numpy_data = {mod: data[mod][0].numpy() for mod in MODALITIES + ['mask']}
            for k, orientation in enumerate(orientations):
                mask_cur_orientation = []
                num_slices = numpy_data[MODALITIES[0]].shape[AXIS_TO_TAKE[k]]
                org_size = tuple([axis_len for axis, axis_len in enumerate(numpy_data[MODALITIES[0]].shape) if
                                  axis != AXIS_TO_TAKE[k]])
                interpolation = cv2.INTER_LINEAR
                ratio = 1.0 if fast_eval else data['ratios'][orientation]
                for j in range(num_slices):
                    pad_data = {}
                    for modality in MODALITIES:
                        slice_modality = slice_with_neighborhood(numpy_data[modality], AXIS_TO_TAKE[k], j,
                                                             opt_test.input_nc // 2, ratio)
                        pad_data[modality] = torch.unsqueeze(transforms.ToTensor()(slice_modality).float(), 0)
                    pad_data['mask'] = torch.zeros_like(pad_data[MODALITIES[0]])
                    # pad_data['mask'] = torch.zeros_like(data[MODALITIES[0]][orientation][j])
                    # for modality in MODALITIES:
                    #     pad_data[modality] = data[modality][orientation][j]

                    slice_all_models = 0
                    for m, current_model in enumerate(models):
                        current_model.set_input({mod: pad_data[mod] for mod in MODALITIES + ['mask']})
                        current_model.test()
                        current_visuals = current_model.get_current_visuals()
                        weight_this_model = 1 if models_weight is None else models_weight[m]
                        slice_this_model = np.squeeze(current_visuals['fake_mask'][0].cpu().numpy())
                        slice_all_models += slice_this_model * weight_this_model
                    numerator = len(models) if models_weight is None else np.sum(models_weight)
                    slice_all_models = np.array(slice_all_models) / numerator
                    slice_all_models = np.squeeze(slice_all_models + 1) / 2
                    # if j == num_slices // 2:
                    #     print(k, orientation, num_slices, j)
                    #     plt.imshow(np.squeeze(pad_data['t1'].cpu().numpy())[1, :, :])
                    #     plt.show()
                    if not fast_eval:
                        slice_all_models = cv2.resize(slice_all_models, (org_size[1], org_size[0]), interpolation)
                    mask_cur_orientation.append(slice_all_models)

                mask_pred += np.transpose(np.squeeze(mask_cur_orientation), transpose[AXIS_TO_TAKE[k]]) * \
                             orientation_weight[k]
            tmp = np.array(mask_pred) / np.sum(orientation_weight)
            mask_pred = np.zeros(data['org_size']['all'])
            bbox = data['bbox']
            mask_pred[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = tmp
            dict_results[hash_label] = mask_pred

            alt_image = nib.load(alt_path)
            if save_membership:
                mask_membership_name = alt_path.replace('%s.%s' % (alt_modality, SUFFIX),
                                                        'membership_%s.%s' % (mask_suffix, SUFFIX))
                nib.Nifti1Image(mask_pred, alt_image.affine, alt_image.header).to_filename(mask_membership_name)
        else:
            mask_pred = dict_results[hash_label]

        mask_pred = (mask_pred > 0.5).astype(np.int8)
        seg_labels, seg_num = measure.label(mask_pred, return_num=True, connectivity=2)

        small_cnt, large_cnt = 0, 0
        for label in range(1, seg_num + 1):
            pixel_cnt = np.sum(mask_pred[seg_labels == label])
            if pixel_cnt <= opt_test.lesion_thres:
                mask_pred[seg_labels == label] = 0
                small_cnt += 1
            else:
                large_cnt += 1

        mask_pred_name = alt_path.replace('%s.%s' % (alt_modality, SUFFIX), 'pred_%s.%s' % (mask_suffix, SUFFIX))
        nib.Nifti1Image(mask_pred, alt_image.affine, alt_image.header).to_filename(mask_pred_name)

        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata().astype(np.int8)
            res_this_mask = {}
            if opt_test.use_anima:
                if np.sum(mask_data):
                    anima_seg_perf = os.path.join(opt_test.anima_path, 'animaSegPerfAnalyzer')
                    cmd = f"{anima_seg_perf} -o {hash_label} -r {mask_path} -i {mask_pred_name} -s -l -d -X"
                    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
                    process.wait()
                    tree = ET.parse(f'{hash_label}_global.xml')
                    root = tree.getroot()
                    for elem in root:
                        res_this_mask[elem.attrib['name'].lower()] = np.float(elem.text)
                    res_this_mask['f1_score'] = 2 * res_this_mask['sensl'] * res_this_mask['ppvl'] / (
                                res_this_mask['ppvl'] + res_this_mask['sensl'] + 1e-6)
                    res_this_mask['f2_score'] = 5 * res_this_mask['sensl'] * res_this_mask['ppvl'] / (
                            4 * res_this_mask['ppvl'] + res_this_mask['sensl'] + 1e-6)
                    if np.isnan(res_this_mask['f1_score']):
                        res_this_mask['f1_score'] = 0.0
                else:
                    res_this_mask['empty_lesion_number'] = large_cnt
                    res_this_mask['empty_lesion_volume'] = np.sum(mask_pred)
                for metric in res_this_mask.keys():
                    ret_metrics[metric].append(res_this_mask[metric])
            else:
                res_this_mask = seg_metrics(mask_pred, mask_data, output_errors=False, fast_eval=fast_eval,
                                            cumulative=opt_test.cumulative, rm_empty=opt_test.rm_empty)
                metrics = list(res_this_mask.keys())
                for k in metrics:
                    ret_metrics[k].append(res_this_mask[k])
            print_metrics('processed ' + basename + '*,', res_this_mask)
        else:
            print('processed ' + basename + '*')

        if not save_images and os.path.exists(mask_pred_name):
            os.remove(mask_pred_name)

    print("time used for validation: ", time.time() - start_time)
    if opt_test.cumulative:
        return seg_metrics_cumulative(ret_metrics)
    for k in list(ret_metrics.keys()):
        if opt_test.rm_empty:
            ret_metrics[k] = np.array(ret_metrics[k])
            ret_metrics[k] = ret_metrics[k][np.isfinite(ret_metrics[k])]
        ret_mean[k] = np.mean(ret_metrics[k]) if num_test != 0 else 0
        ret_std[k] = np.std(ret_metrics[k]) if num_test != 0 else 0
    if 'dice' in ret_mean and 'f1_score' in ret_mean:
        ret_mean['msseg_score'] = ret_mean['dice'] + ret_mean['f1_score'] - ret_mean['empty_lesion_number'] / 200.0 \
                                  - ret_mean['empty_lesion_volume'] / 4000.0
    print_metrics('std ', ret_std)
    return ret_mean


if __name__ == '__main__':
    opt_test = TestOptions().parse()

    # hard-code some parameters for test
    opt_test.num_threads = 0   # test code only supports num_threads = 1
    opt_test.batch_size = 1    # test code only supports batch_size = 1
    opt_test.serial_batches = True  # no shuffle
    opt_test.no_flip = True    # no flip
    opt_test.display_id = -1   # no visdom display
    opt_test.dataset_mode = 'ms_3d'
    data_loader = CreateDataLoader(opt_test)
    dataset_test = data_loader.load_data()

    models = []
    models_indx = opt_test.load_str.split(',')
    models_weight = [1] * len(models_indx)
    for i in models_indx:
        current_model = create_model(opt_test, i)
        current_model.setup(opt_test)
        if opt_test.eval:
            current_model.eval()
        models.append(current_model)

    mean = model_test(models, dataset_test, opt_test, len(data_loader), save_images=True,
                       models_weight=models_weight, mask_suffix=opt_test.name, save_membership=True)
    print_metrics('mean ', mean)
