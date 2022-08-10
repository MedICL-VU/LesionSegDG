import pandas as pd
import numpy as np
import os


def output_prev_results():
    single_names = ["new_env", "working", "isbi_rotate_transpose_distort_resize", "isbi_basic", "isbi_basic_1",
                 "isbi_basic_0511", "isbi_reproduce_baseline_0517", "isbi_reproduce_baseline_0814",
                 "isbi_reproduce_baseline_0817", "isbi_reproduce_baseline_accre1", "isbi_reproduce_baseline_accre2"]
    mult_names = ["isbi_reproduce_baseline", "isbi_reproduce_baseline_randcrop", "isbi_reproduce_baseline_randcrop_ratio"]
    suffix = ['val0', 'val1',  'val2', 'val3','val4']   # 'val1', '12', '13', '14', '15', '16'
    for n in single_names+mult_names:
        res = []
        res1 = []
        valid_suffix = []
        for s in suffix:
            appendix = "_0" if n in mult_names else ""
            f = "/home/huahong/Documents/Checkpoints/%s/val_loss_log_%s%s.csv" % (n, s, appendix)
            if not os.path.exists(f):
                continue
            cur_csv = pd.read_csv(f)
            score = (1-cur_csv['lfpr']) / 4 + cur_csv['ltpr'] / 4 +\
                        cur_csv['corr'] / 4 + cur_csv['ppv'] / 8 + cur_csv['dice'] / 8
            res.append(score[cur_csv['dice'].idxmax()])
            valid_suffix.append(s)
            f1 = "/home/huahong/Documents/Checkpoints/%s/val_loss_log_%s_1.csv" % (n, s)
            if not os.path.exists(f1):
                continue
            cur_csv1 = pd.read_csv(f1)
            score1 = (1-cur_csv1['lfpr']) / 4 + cur_csv1['ltpr'] / 4 +\
                        cur_csv1['corr'] / 4 + cur_csv1['ppv'] / 8 + cur_csv1['dice'] / 8
            res1.append(score1[cur_csv['dice'].idxmax()])
        print(n, res, np.array(res).mean(), valid_suffix)
        if res1:
            print(res1, np.array(res1).mean())


if __name__ == '__main__':
    output_prev_results()