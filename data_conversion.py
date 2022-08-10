import os
import shutil
import nibabel as nib
import numpy as np
import json
from util.util import mkdir
from util.image_property import normalize_image, hash_file


def get_ids(dataset_config):
    dic_ids = {}
    path_dataset = os.path.expanduser(dataset_config['path'])
    mkdir(os.path.join(path_dataset, 'raw'))
    fnames = sorted(os.listdir(path_dataset))
    for fname in fnames:
        if not fname.endswith(dataset_config['suffix']):
            continue
        fname_tmp = fname.split('.')[0]
        prefix, patient_str, timepoint_str, modality = fname_tmp.split('_')
        patient_id, timepoint_id = int(patient_str), int(timepoint_str)
        if patient_id in dic_ids and timepoint_id in dic_ids[patient_id]:
            continue

        # if there is new patient id and timepoint id, we get all the modalities and masks based on the constants
        # and we will move the files into the 'raw' subdirectory
        if patient_id not in dic_ids:
            dic_ids[patient_id] = {}
        dic_ids[patient_id][timepoint_id] = {'modalities':{}, 'mask':{}}
        for mod in dataset_config['modalities']+dataset_config['masks']:
            fname_modality = '_'.join((prefix, patient_str, timepoint_str, mod)) + '.' + dataset_config['suffix']
            path_modality_src = os.path.join(path_dataset, fname_modality)
            path_modality_dst = os.path.join(path_dataset, 'raw', fname_modality)
            category = 'modalities' if mod in dataset_config['modalities'] else 'mask'
            assert os.path.exists(path_modality_src)
            shutil.move(path_modality_src, path_modality_dst)
            dic_ids[patient_id][timepoint_id][category][mod] = path_modality_dst
    fname_json = os.path.join(path_dataset, 'ids.json')
    with open(fname_json, 'w') as f:
        json.dump(dic_ids, f, indent=2)
    return dic_ids


def get_properties(dataset_path):
    fname_json = os.path.join(dataset_path, 'ids.json')
    with open(fname_json, 'r') as f:
        dic_ids = json.load(f)
    dic_properties = {}
    for patient_id in dic_ids:
        for timepoint_id in dic_ids[patient_id]:
            for modality in dic_ids[patient_id][timepoint_id]['modalities']:
                path_modality = dic_ids[patient_id][timepoint_id]['modalities'][modality]
                label = hash_file(path_modality)
                data = nib.load(path_modality).get_fdata()
                peak = normalize_image(data, modality)
                peak = peak[0] if isinstance(peak, np.ndarray) else peak
                dic_properties[label] = {}
                dic_properties[label]['path'] = path_modality
                dic_properties[label]['peak'] = peak
    fname_json = os.path.join(dataset_path, 'properties.json')
    with open(fname_json, 'w') as f:
        json.dump(dic_properties, f, indent=2)
    return dic_properties


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    for dataset in config:
        dataset_config = config[dataset]
        assert os.path.exists(dataset_config['path'])

        dic_ids = get_ids(dataset_config)
        dic_properties = get_properties(dataset_config['path'])
