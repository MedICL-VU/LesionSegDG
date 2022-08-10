import time
import copy
import os
import torch
import random
import numpy as np
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from data.data_generator import DataGenerator
from data.data_builder import DataBuilder
import test


def set_seed():
    seed = 10
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # make cudnn to be reproducible for performance
    # can be commented for faster training
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_val_test_opts(opt_train):
    opt_val = copy.deepcopy(opt_train)
    opt_val.phase = 'val'
    opt_val.num_threads = 0
    opt_val.batch_size = 1
    opt_val.serial_batches = True  # no shuffle
    opt_val.no_flip = True  # no flip
    opt_val.dataset_mode = 'ms_3d'

    opt_test = copy.deepcopy(opt_val)
    opt_test.phase = 'test'
    return opt_val, opt_test


def create_data_loader(opt_this_phase):
    data_loader = CreateDataLoader(opt_this_phase)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#%s images = %d' % (opt_this_phase.phase, dataset_size))
    return dataset, dataset_size


if __name__ == '__main__':
    set_seed()
    print('process id ', os.getpid())

    opt = TrainOptions().parse()
    opt_val, opt_test = get_val_test_opts(opt)
    test_index = opt.n_fold - 1 if opt.test_index is None else opt.test_index
    test_index = opt.n_fold if 'test' not in opt.test_mode else test_index
    val_indices = [x for x in range(opt.n_fold) if x != test_index] if opt.test_mode != 'test' else [test_index]
    val_indices = val_indices if opt.start_index is None else [x for x in range(opt.start_index, opt.n_fold) if x != test_index]
    val_indices = val_indices if opt.val_index is None else [opt.val_index]
    best_metrics = opt.best_metrics.split(',')

    models = []
    # data_generator = DataGenerator(opt.dataroot)
    data_builder = DataBuilder(opt)
    for val_index in val_indices:
        data_builder.build_dataset(val_index, test_index)
        source_idx, target_idx, dataset_domain = data_builder.get_dataset_type()
        # data_generator.build_dataset(val_index, test_index, opt.test_mode)
        dataset, dataset_size = create_data_loader(opt)
        dataset_val, dataset_size_val = [], []
        all_opt_val = data_builder.get_multiple_opt(opt_val)
        for this_opt_val in all_opt_val:
            this_dataset_val, this_dataset_size_val = create_data_loader(this_opt_val)
            dataset_val.append(this_dataset_val)
            dataset_size_val.append(this_dataset_size_val)

        model_suffix = 'val%d' % val_index if 'val' in opt.test_mode else ''
        model_suffix += 'test%d' % test_index if 'test' in opt.test_mode else ''
        model = create_model(opt, model_suffix)
        model.setup(opt)
        visualizer = Visualizer(opt)

        total_steps, best_epochs, val_losses_best = 0, 0, 0
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
                model.save_networks(epoch)

            if epoch % opt.val_epoch_freq == 0:
                start_time_val = time.time()
                if opt_val.eval_val:
                    model.eval()
                    model.isTrain = False
                losses_val = []
                for i, this_opt_val in enumerate(all_opt_val):
                    if dataset_size_val[i] <= 0:
                        continue
                    this_losses_val = test.model_test([model], dataset_val[i], this_opt_val, dataset_size_val[i])
                    if opt.display_id > 0:
                        visualizer.plot_val_losses(epoch, 0, this_opt_val, this_losses_val, dataset_num=i,
                                                   model_suffix=model_suffix)
                    else:
                        visualizer.save_val_losses(epoch, 0, this_opt_val, this_losses_val, dataset_num=i,
                                                   model_suffix=model_suffix)
                    visualizer.print_val_losses(epoch, this_losses_val, time.time() - start_time_val)
                    losses_val.append(this_losses_val)
                model.train()
                model.isTrain = True

                cur_loss_val = 0
                for i in source_idx:
                    for metric in best_metrics:
                        cur_loss_val += losses_val[i][metric]

                if cur_loss_val > val_losses_best:
                    val_losses_best = cur_loss_val
                    best_epochs = epoch
                    model.save_networks('latest')
                elif epoch - best_epochs >= 400 and 'val' in opt.test_mode:
                    break

                print("best epoch", best_epochs, "best loss", val_losses_best)

            print('finished epoch %d / %d, \t time taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
        models.append(model)

    dataset_test, dataset_size_test = [], []
    all_opt_test = data_builder.get_multiple_opt(opt_test)
    for this_opt_test in all_opt_test:
        this_dataset_val, this_dataset_size_val = create_data_loader(this_opt_test)
        dataset_test.append(this_dataset_val)
        dataset_size_test.append(this_dataset_size_val)
    losses_test = []
    for i, this_opt_val in enumerate(all_opt_test):
        this_losses_test = test.model_test(models, dataset_test[i], opt_test, dataset_size_test[i], save_images=True,
                                      mask_suffix=opt_test.name, save_membership=True)
        losses_test.append(this_losses_test)
    print(losses_test)
