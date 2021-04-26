from .DataCfgBase import DataCfgBase
from batchgenerators.utilities.file_and_folder_operations import subfiles
from copy import deepcopy
import numpy as np


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "border_mode_data": "constant",

    "num_threads": 12,
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)

default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True


class SynapseAbdomenCfg(object):
    def __init__(self):
        super().__init__()
        self.img_data_root = '/home/xxx/DataSet/Synapse_Multi_Organ_Abdomen/Abdomen/RawData/Training/img'
        self.lbl_data_root = '/home/xxx/DataSet/Synapse_Multi_Organ_Abdomen/Abdomen/RawData/Training/label'
        self.img_fname_list, self.lbl_fname_list = self.__data_list(join=False)
        self.img_path_list, self.lbl_path_list = self.__data_list(join=True)

        self.data_modal = 'CT'

        # dataset
        self.three_dim = False
        self.z_first = False     # set (x, y, z) to (z, x, y)
        self.load_into_memory = True
        self.is_test = False
        if self.lbl_data_root is None:  # test mode
            self.is_test = True

        # dataloader
        self.patch_size = (512, 512)
        self.batch_size = 16
        self.num_threads_in_multithreaded = 1
        self.seed_for_shuffle = None
        self.return_incomplete = False  # if len(data_size) % batch_size > 0, whether to return the left incomplete data batch, default is to discard the left
        self.shuffle = True
        self.infinite = False
        if self.lbl_data_root is None:  # test mode
            self.shuffle = False
            self.return_incomplete = True

        # data augmentation
        if self.three_dim:
            self.augmentation_params = default_3D_augmentation_params
        else:
            self.augmentation_params = default_2D_augmentation_params

    def __data_list(self, join):
        '''
        :param join: whether join the root path, if False, only return filename
        :return: img_file_list, lbl_file_list, sorted by name
        '''
        if self.lbl_data_root is None:
            return subfiles(self.img_data_root, join), None
        return subfiles(self.img_data_root, join), subfiles(self.lbl_data_root, join)
