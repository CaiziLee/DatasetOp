import numpy as np
from batchgenerators.dataloading import Dataset
from medpy.io import load
from DataCfg.VarDefine import *


class SynapseAbdomenDataset(Dataset):
    def __init__(self, configs):
        super(SynapseAbdomenDataset, self).__init__()
        self.__data_fname_list, self.__lbl_fname_list = configs.img_fname_list, configs.lbl_fname_list  # sorted
        self.__data_path_list, self.__lbl_path_list = configs.img_path_list, configs.lbl_path_list  # sorted
        self.__is_test = configs.is_test

        self.__z_first = configs.z_first  # set (x, y, z) to (z, x, y), default is True
        self.__load_into_memory = configs.load_into_memory  # whether to load all dataset into memory
        self.__three_dim = configs.three_dim  # whether to load dataset in 3D

        self.__data = {}  # data dict
        self.__slice_list = []  # 2D: slices_ids    [img00001.nii.gz_0, img00001.nii.gz_1, ...]

        for fname in self.__data_fname_list:
            self.__data[fname] = {}  # initialize data dict
        if self.__load_into_memory:  # load the whole dataset into memory, will be faster.
            self.__load_data_into_memory()  # fill self.__data
        else:  # if the dataset is too big enough
            self.__load_data_info()  # only load data info, such as path etc.

        self.__item_list = self.__data_fname_list  # 3D item list  [img00001.nii.gz, img00002.nii.gz, ...]

    def __assemble_slice_name(self, img_id, slice_id):  # img00001.nii.gz_0
        return str(img_id) + '_' + str(slice_id)

    def __disassemble_slice_name(self, slice_name):
        return slice_name[: slice_name.rindex('_')], int(slice_name[slice_name.rindex('_') + 1:])

    def __load_data_into_memory(self):
        for idx in range(len(self.__data_path_list)):
            item_name = self.__data_fname_list[idx]
            data_path = self.__data_path_list[idx]
            img, header = load(data_path)  # (512, 512, 100)
            self.__data[item_name][IMG] = np.moveaxis(img, -1, 0) if self.__z_first else img
            self.__data[item_name][HEADER] = header
            self.__data[item_name][NB_SLICES] = img.shape[-1]
            self.__data[item_name][IMG_PATH] = data_path
            if self.__is_test is False:
                lbl_path = self.__lbl_path_list[idx]
                lbl, lheader = load(lbl_path)
                self.__data[item_name][LABEL] = np.moveaxis(lbl, -1, 0) if self.__z_first else lbl
                self.__data[item_name][LABEL_PATH] = lbl_path

            for slice_id in range(img.shape[-1]):
                self.__slice_list.append(self.__assemble_slice_name(item_name, slice_id))

    def __load_data_info(self):
        for idx in range(len(self.__data_path_list)):
            item_name = self.__data_fname_list[idx]
            data_path = self.__data_path_list[idx]
            img, header = load(data_path)  # (512, 512, 100)
            self.__data[item_name][NB_SLICES] = img.shape[-1]
            self.__data[item_name][IMG_PATH] = data_path
            if self.__is_test is False:
                lbl_path = self.__lbl_path_list[idx]
                self.__data[item_name][LABEL_PATH] = lbl_path
            for slice_id in range(img.shape[-1]):
                self.__slice_list.append(self.__assemble_slice_name(item_name, slice_id))

    def __getitem__(self, item):
        '''
        needs to return a data_dict for the sample at the position item;
        :param item:
        :return:
        '''
        if self.__load_into_memory:
            if self.__three_dim:
                if self.__is_test is False:
                    return {IMG: self.__data[self.__item_list[item]][IMG].astype(np.float32),
                            LABEL: self.__data[self.__item_list[item]][LABEL].astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: self.__item_list[item],
                                IMG_PATH: self.__data[self.__item_list[item]][IMG_PATH],
                                LABEL_PATH: self.__data[self.__item_list[item]][LABEL_PATH],
                                NB_SLICES: self.__data[self.__item_list[item]][NB_SLICES],
                                HEADER: self.__data[self.__item_list[item]][HEADER],
                            }, }
                else:
                    return {IMG: self.__data[self.__item_list[item]][IMG].astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: self.__item_list[item],
                                IMG_PATH: self.__data[self.__item_list[item]][IMG_PATH],
                                NB_SLICES: self.__data[self.__item_list[item]][NB_SLICES],
                                HEADER: self.__data[self.__item_list[item]][HEADER],
                            }, }
            else:  # 2D, retuen slice
                item_name, slice_id = self.__disassemble_slice_name(self.__slice_list[item])
                if self.__z_first:
                    img = self.__data[item_name][IMG][slice_id].astype(np.float32)
                    if self.__is_test is False:
                        lbl = self.__data[item_name][LABEL][slice_id].astype(np.float32)
                else:
                    img = self.__data[item_name][IMG][:, :, slice_id].astype(np.float32)
                    if self.__is_test is False:
                        lbl = self.__data[item_name][LABEL][:, :, slice_id].astype(np.float32)
                if self.__is_test is False:
                    return {IMG: img,
                            LABEL: lbl,
                            PROPERTIES: {
                                IMG_ID: item_name,
                                SLICE_ID: slice_id,
                                IMG_PATH: self.__data[item_name][IMG_PATH],
                                LABEL_PATH: self.__data[item_name][LABEL_PATH],
                                NB_SLICES: self.__data[item_name][NB_SLICES],
                                HEADER: self.__data[item_name][HEADER],
                            }, }
                else:
                    return {IMG: img,
                            PROPERTIES: {
                                IMG_ID: item_name,
                                SLICE_ID: slice_id,
                                IMG_PATH: self.__data[item_name][IMG_PATH],
                                NB_SLICES: self.__data[item_name][NB_SLICES],
                                HEADER: self.__data[item_name][HEADER],
                            }, }
        else:
            if self.__three_dim:
                img_path, lbl_path = self.__data[self.__item_list[item]][IMG_PATH], self.__data[self.__item_list[item]][LABEL_PATH]
                img, header = load(img_path)  # (512, 512, 100)
                lbl, lheader = load(lbl_path)
                if self.__is_test is False:
                    return {IMG: np.moveaxis(img.astype(np.float32), -1, 0) if self.__z_first else img.astype(np.float32),
                            LABEL: np.moveaxis(lbl.astype(np.float32), -1, 0) if self.__z_first else lbl.astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: self.__item_list[item],
                                IMG_PATH: self.__data[self.__item_list[item]][IMG_PATH],
                                LABEL_PATH: self.__data[self.__item_list[item]][LABEL_PATH],
                                NB_SLICES: self.__data[self.__item_list[item]][NB_SLICES],
                                HEADER: header,
                            }, }
                else:
                    return {IMG: np.moveaxis(img.astype(np.float32), -1, 0) if self.__z_first else img.astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: self.__item_list[item],
                                IMG_PATH: self.__data[self.__item_list[item]][IMG_PATH],
                                NB_SLICES: self.__data[self.__item_list[item]][NB_SLICES],
                                HEADER: header,
                            }, }
            else:
                item_name, slice_id = self.__disassemble_slice_name(self.__slice_list[item])
                img_path, lbl_path = self.__data[item_name][IMG_PATH], self.__data[item_name][LABEL_PATH]
                img, header = load(img_path)  # (512, 512, 100)
                lbl, lheader = load(lbl_path)
                if self.__is_test is False:
                    return {IMG: img[:, :, slice_id].astype(np.float32) if self.__z_first else img[slice_id].astype(np.float32),
                            LABEL: lbl[:, :, slice_id].astype(np.float32) if self.__z_first else lbl[slice_id].astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: item_name,
                                SLICE_ID: slice_id,
                                IMG_PATH: self.__data[item_name][IMG_PATH],
                                LABEL_PATH: self.__data[item_name][LABEL_PATH],
                                NB_SLICES: self.__data[item_name][NB_SLICES],
                                HEADER: header,
                            }, }
                else:
                    return {IMG: img[:, :, slice_id].astype(np.float32) if self.__z_first else img[slice_id].astype(np.float32),
                            PROPERTIES: {
                                IMG_ID: item_name,
                                SLICE_ID: slice_id,
                                IMG_PATH: self.__data[item_name][IMG_PATH],
                                NB_SLICES: self.__data[item_name][NB_SLICES],
                                HEADER: header,
                            }, }

    def __len__(self):
        '''
        returns how many items the dataset has
        :return:
        '''
        if self.__three_dim:  # 3D image as an item
            return len(self.__item_list)
        else:
            return len(self.__slice_list)


# from DataCfg.DataCfg import SynapseAbdomenCfg
# a = SynapseAbdomenCfg()
# dataset = SynapseAbdomenDataset(a)
# print(len(dataset))
# item = dataset[10]
# print(type(item), type(item[IMG]), item[IMG].dtype)
# import matplotlib.pyplot as plt
# plt.imshow(item[IMG])
# plt.show()
# print(item[LABEL].shape, item[LABEL].shape)
# print(img.shape, lbl.shape)
# a = SynapseAbdomen(channel_first=False)
# data = a.load()
# print(data['img0030.nii.gz']['img'].shape, data['img0030.nii.gz']['nb_slice'])
