from batchgenerators.dataloading import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image, random_crop_2D_image_batched, random_crop_3D_image_batched, \
    random_crop_3D_image, random_crop_2D_image
from DataCfg.VarDefine import *
import numpy as np


class CustomDataLoader(DataLoader):
    '''
    only accept Dataset type
    can handle 2D or 3D dataset
    '''
    def __init__(self, configs, data):
        super(CustomDataLoader, self).__init__(data, configs.batch_size, num_threads_in_multithreaded=configs.num_threads_in_multithreaded,
                                                   seed_for_shuffle=configs.seed_for_shuffle, return_incomplete=configs.return_incomplete,
                                                   shuffle=configs.shuffle, infinite=configs.infinite)
        self.indices = np.arange(len(data))
        self.cfg = configs

    def generate_train_batch(self):
        indices = self.get_indices()
        imgs = []
        labels = []
        properties = []
        for idx in indices:
            item = self._data[idx]
            if self.cfg.is_test is False:
                img, lbl, _property = item[IMG], item[LABEL], item[PROPERTIES]
                stacked_volume = np.stack([img, lbl])   # (2, 512, 512)

                assert len(img.shape) == len(self.cfg.patch_size), "len(patch_size) must be equal to len(img.shape)"

                padded_stacked_volume = pad_nd_image(stacked_volume, self.cfg.patch_size)   # in case the img_size is smaller than patch_size
                padded_stacked_volume = np.expand_dims(padded_stacked_volume, axis=0)   # (1, 2, *size)
                if self.cfg.three_dim:
                    cropped_stacked_volume = random_crop_3D_image_batched(padded_stacked_volume, self.cfg.patch_size)
                else:
                    cropped_stacked_volume = random_crop_2D_image_batched(padded_stacked_volume, self.cfg.patch_size)
                cropped_stacked_volume = np.squeeze(cropped_stacked_volume)     # (2, *patch_size)
                img, lbl = cropped_stacked_volume[0], cropped_stacked_volume[1]
                imgs.append(img)
                labels.append(lbl)
                properties.append(_property)
            else:
                img, _property = item[IMG], item[PROPERTIES]

                assert len(img.shape) == len(self.cfg.patch_size), "len(patch_size) must be equal to len(img.shape)"

                padded_stacked_volume = pad_nd_image(img, self.cfg.patch_size)  # in case the img_size is smaller than patch_size
                if self.cfg.three_dim:
                    cropped_stacked_volume = random_crop_3D_image(padded_stacked_volume, self.cfg.patch_size)
                else:
                    cropped_stacked_volume = random_crop_2D_image(padded_stacked_volume, self.cfg.patch_size)
                imgs.append(cropped_stacked_volume)
                properties.append(_property)

        batch_img = np.expand_dims(np.stack(imgs), axis=1)  # (b, c, *patch_size)
        if self.cfg.is_test:
            return {IMG: batch_img, BATCH_KEYS: indices, PROPERTIES: properties}
        batch_label = np.expand_dims(np.stack(labels), axis=1)  # (b, c, *patch_size)
        return {IMG: batch_img, LABEL: batch_label, BATCH_KEYS: indices, PROPERTIES: properties}


# from DataCfg.DataCfg import SynapseAbdomenCfg
# from Dataset.SynapseAbdomenDataset import SynapseAbdomenDataset
# import matplotlib.pyplot as plt
# cfg = SynapseAbdomenCfg()
# dataset = SynapseAbdomenDataset(cfg)
# dloader = CustomDataLoader(cfg, dataset)
# for batch in dloader:
#     a = batch
#     print(a[IMG].shape, a[LABEL].shape, a[BATCH_KEYS], len(a[PROPERTIES]), a[PROPERTIES][0])
#     # print(a[IMG].shape, a[BATCH_KEYS], len(a[PROPERTIES]), a[PROPERTIES][0])
#     plt.subplot(221), plt.imshow(a[IMG][1, 0, :, :])
#     plt.subplot(222), plt.imshow(a[LABEL][1, 0, :, :])
#     plt.show()
