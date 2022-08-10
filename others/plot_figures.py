import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import cv2
from matplotlib import colors, cm


def plot_figures():
    base_dir = '/home/huahong/Documents/Datasets/'
    params = [
        [1, 1, 1],
        ['08002CHJE', 3, 1],  # also OK: 3, 5, 12
        [1, 1]
    ]
    images_paths = {
        'isbi': {
            # 't1': base_dir + 'isbi_dataset/raw/training_%02d_%02d_t1.nii' % (params[0][0], params[0][1]),
            # 'flair':base_dir + 'isbi_dataset/raw/training_%02d_%02d_flair.nii' % (params[0][0], params[0][1]),
            # 't2': base_dir + 'isbi_dataset/raw/training_%02d_%02d_t2.nii' % (params[0][0], params[0][1]),
            # 'pd': base_dir + 'isbi_dataset/raw/training_%02d_%02d_pd.nii' % (params[0][0], params[0][1]),
            # 'mask':base_dir + 'isbi_dataset/raw/training_%02d_%02d_mask%d.nii' % (params[0][0], params[0][1], params[0][2]),
            't1': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_t1.nii' % (params[0][0], params[0][1]),
            'flair': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_flair.nii' % (params[0][0], params[0][1]),
            't2': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_t2.nii' % (params[0][0], params[0][1]),
            'pd': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_pd.nii' % (params[0][0], params[0][1]),
            'mask': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_mask.nii' % (params[0][0], params[0][1]),
            'pred_isbi_bl': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_pred_colab_isbi_3mod_baselineval0.nii' %
                       (params[0][0], params[0][1]),
            'pred_isbi_aug3': base_dir + 'isbi_dataset/val_all/%d_mask1_val%d_pred_colab_isbi_3mod_aug3val0.nii' %
                            (params[0][0], params[0][1]),
        },
        'msseg': {
            't1': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_t1.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            'flair': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_flair.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            't2': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_t2.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            'pd': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_pd.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            'ce': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_ce.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            'mask': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_mask.nii.gz' % (params[1][0], params[1][1], params[1][2]),
            'pred_isbi_bl': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_pred_colab_isbi_3mod_baselineval0.nii.gz' %
                       (params[1][0], params[1][1], params[1][2]),
            'pred_isbi_aug3': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_pred_colab_isbi_3mod_aug3val0.nii.gz' %
                            (params[1][0], params[1][1], params[1][2]),
        },
        'lit': {
            't1': base_dir + 'lit_dataset/raw/patient_%02d_%02d_t1.nii.gz' % (params[2][0], params[2][1]),
            'flair': base_dir + 'lit_dataset/raw/patient_%02d_%02d_flair.nii.gz' % (params[2][0], params[2][1]),
            't2': base_dir + 'lit_dataset/raw/patient_%02d_%02d_t2.nii.gz' % (params[2][0], params[2][1]),
            'ce': base_dir + 'lit_dataset/raw/patient_%02d_%02d_ce.nii.gz' % (params[2][0], params[2][1]),
            'mask': base_dir + 'lit_dataset/raw/patient_%02d_%02d_mask.nii.gz' % (params[2][0], params[2][1]),
            'pred_isbi_bl': base_dir + 'lit_dataset/raw/patient_%02d_%02d_pred_colab_isbi_3mod_baselineval0.nii.gz' % (params[2][0], params[2][1]),
            'pred_isbi_aug3': base_dir + 'lit_dataset/raw/patient_%02d_%02d_pred_colab_isbi_3mod_aug3val0.nii.gz' % (
                        params[2][0], params[2][1]),
        }
    }

    axis_to_take = {
        'isbi': 2,
        'msseg': 2,
        'lit': 2
    }

    fig = plt.figure(1, figsize=(10, 5.0))

    mods = ['t1', 'flair', 't2', 'pd', 'ce', 'mask', 'pred_isbi_bl', 'pred_isbi_aug3']
    cnt = 0
    xlabels = ['T1-w', 'FLAIR', 'T2-w', 'PD', 'CE', 'Baseline', 'Ours']
    ylabels = ['Source: ISBI', 'Target: MICCAI16', 'Target: UMCL']
    height, width = 200, 200
    # pos_x_y = [(7, 18), (9, 26), (7, 20)]
    for i, k in enumerate(images_paths.keys()):
        data = {}
        ratio = 1.0
        bbox = None
        sum_axes = tuple(a for a in range(3) if a != axis_to_take[k])
        sl = [slice(None), slice(None), slice(None)]

        gt, flair = None, None
        for j, mod in enumerate(mods):
            if mod not in images_paths[k]:
                continue
            image = nib.load(images_paths[k][mod])
            data[mod] = image.get_data()
            if mod == 'mask':
                max_mask = np.sum(data[mod], axis=sum_axes)
                idx = np.argmax(max_mask)
                sl[axis_to_take[k]] = slice(idx, idx+1, 1)
                voxel_sizes = nib.affines.voxel_sizes(image.affine)
                ratio = [b for a, b in enumerate(voxel_sizes) if a != axis_to_take[k]]
                ratio = ratio[0] / ratio[1]
                # print(idx, ratio, sl, max_mask)

        for j, mod in enumerate(mods):
            if mod in images_paths[k]:
                slice_image = np.rot90(np.squeeze(data[mod][tuple(sl)]), 3)
                if j == 0 and mod != 'mask':
                    a = np.where(slice_image != 0)
                    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
                slice_image = slice_image[bbox[0]-5: bbox[1]+5, bbox[2]-5: bbox[3]+5]
                height, width = slice_image.shape[:2]
                if abs(ratio - 1.0) > 0.1:
                    if ratio > 1:
                        height, width = height, int(width * ratio)
                    else:
                        height, width = int(height / ratio), width
                    height = int((200 / width) * height)
                    width = 200
                    slice_image = cv2.resize(slice_image, (width, height), cv2.INTER_LINEAR)
            else:
                slice_image = np.ones((height, width))

            if mod == 'flair':
                flair = slice_image.copy()

            if mod == 'mask':
                gt = slice_image.copy()

            if mod.startswith('pred'):
                norm = colors.Normalize()
                cmap = plt.cm.gray
                col1 = colors.colorConverter.to_rgba('g')
                col2 = colors.colorConverter.to_rgba('r')
                col3 = colors.colorConverter.to_rgba('b')
                fl_colors2 = cmap(norm(flair))
                fl_colors2[(gt > 0.99) & (slice_image > 0.99), :] = col1
                fl_colors2[(gt < 0.99) & (slice_image > 0.99), :] = col2  # false positive in red
                fl_colors2[(gt > 0.99) & (slice_image < 0.99), :] = col3  # false negtive in green
                slice_image = fl_colors2

            if mod != 'mask':
                cnt += 1
                ax = fig.add_subplot(len(axis_to_take), len(mods)-1, cnt)
                # plt.text(pos_x_y[i][0], pos_x_y[i][1], chr(ord('A') + i) + '-' + str(j+1), fontsize=9, color='black',
                #          bbox=dict(facecolor='wheat'))
                # plt.axis('off')
                ax.imshow(slice_image, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(xlabels[(cnt - 1) % (len(mods) - 1)])
                if j == 0:
                    ax.set_ylabel(ylabels[i])
                if mod == 'ce' and i == 0:
                    ax.text(0.5, 0.5, 'ISBI\ncontains\nno CE\nimages', color='white', transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')
                if mod == 'pd' and i == 2:
                    ax.text(0.5, 0.5, 'UMCL\ncontains\nno PD\nimages', color='white', transform=ax.transAxes, verticalalignment='center', horizontalalignment='center')

    fig.tight_layout()
    # fig.savefig("example1_isbi_model.pdf")
    fig.show()


def plot_figures2():
    base_dir = '/home/huahong/Documents/Datasets/'
    mods = ['t1', 'flair', 't2', 'pd', 'ce', 'mask']
    params = [
        [],
        # ['08002CHJE', 3, 1],  # also OK: 3, 5, 12
        ['01016SACH', 1, 1]
    ]
    images_paths = {
        't1': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_t1.nii.gz' % (params[1][0], params[1][1], params[1][2]),
        'flair': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_flair.nii.gz' % (params[1][0], params[1][1], params[1][2]),
        't2': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_t2.nii.gz' % (params[1][0], params[1][1], params[1][2]),
        'pd': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_pd.nii.gz' % (params[1][0], params[1][1], params[1][2]),
        'ce': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_ce.nii.gz' % (params[1][0], params[1][1], params[1][2]),
        'mask': base_dir + 'msseg_dataset/raw/%s_%02d_%02d_mask.nii.gz' % (params[1][0], params[1][1], params[1][2]),
    }
    data = {}
    sl = [slice(None), slice(None), slice(None)]
    axis_to_take = 2
    sum_axes = tuple(a for a in range(3) if a != axis_to_take)
    fig = plt.figure(1, figsize=(9.0, 7.0))

    gt, flair = None, None
    ratio = 1
    for j, mod in enumerate(mods):
        if mod not in images_paths:
            continue
        image = nib.load(images_paths[mod])
        data[mod] = image.get_data()
        if mod == 'mask':
            max_mask = np.sum(data[mod], axis=sum_axes)
            idx = np.argmax(max_mask)
            sl[axis_to_take] = slice(idx, idx + 1, 1)
            voxel_sizes = nib.affines.voxel_sizes(image.affine)
            ratio = [b for a, b in enumerate(voxel_sizes) if a != axis_to_take]
            ratio = ratio[0] / ratio[1]

    for j, mod in enumerate(mods):
        slice_image = np.rot90(np.squeeze(data[mod][tuple(sl)]), 3)
        if j == 0 and mod != 'mask':
            a = np.where(slice_image != 0)
            bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        slice_image = slice_image[bbox[0] - 5: bbox[1] + 5, bbox[2] - 5: bbox[3] + 5]
        height, width = slice_image.shape[:2]
        if abs(ratio - 1.0) > 0.1:
            if ratio > 1:
                height, width = height, int(width * ratio)
            else:
                height, width = int(height / ratio), width
            slice_image = cv2.resize(slice_image, (width, height), cv2.INTER_LINEAR)
        ax = fig.add_subplot(2, 3, j + 1)
        plt.text(6, 21, chr(ord('A') + j), fontsize=14, color='black',
                 bbox=dict(facecolor='wheat'))
        ax.imshow(slice_image, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig("msseg2.pdf")
    # fig.show()


if __name__ == '__main__':
    # plot_figures()
    plot_figures2()