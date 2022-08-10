import os
import json
import copy
from data.data_generator import DataGenerator


class DataBuilder:
    def __init__(self, opt):
        self.data_generators = {}
        self.configs = {}
        self.n_fold = opt.n_fold
        self.test_mode = opt.test_mode

        with open('config.json', 'r') as f:
            config = json.load(f)

        for dataset in config:
            dataset_config = config[dataset]
            dataset_path = os.path.expanduser(dataset_config['path'])
            self.data_generators[dataset] = DataGenerator(dataset_path)
            self.configs[dataset] = dataset_config
            assert os.path.exists(dataset_path)

    def build_dataset(self, val_index, test_index):
        # generate train/val/test data using each dataset
        for dataset in self.data_generators:
            print("dataset {}".format(dataset))
            data_generator = self.data_generators[dataset]
            generate_training = self.configs[dataset]['train']
            if generate_training:
                data_generator.build_dataset(val_index, test_index, self.n_fold, 'train_' + self.test_mode,
                                             self.configs[dataset]["suffix"])
            else:
                data_generator.build_dataset(val_index, test_index, self.n_fold, self.test_mode,
                                             self.configs[dataset]["suffix"])

    def get_dataset_type(self):
        source_idx, target_idx = [], []
        dataset_domain = {}
        for i, dataset in enumerate(self.configs):
            dataset_domain[dataset] = self.configs[dataset]['domain']
            if dataset_domain[dataset] == 'source':
                source_idx.append(i)
            elif dataset_domain[dataset] == 'target':
                target_idx.append(i)
            else:
                raise TypeError("dataset domain error")
        return source_idx, target_idx, dataset_domain

    def get_multiple_opt(self, opt):
        all_opt = []
        for dataset in self.data_generators:
            this_path = self.data_generators[dataset].get_info()["dataroot"]
            all_opt.append(copy.deepcopy(opt))
            all_opt[-1].dataroot = this_path
        return all_opt
