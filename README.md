# DatasetOp
This is an easy-to-use dataset operation project, which can be used in medical image analysis.

It contains four parts, [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) must be installed first, part of codes comes from [nnUNet](https://github.com/MIC-DKFZ/nnUNet), thanks to Fabian Isensee et al. for their contributions: 
- DatasetCfg: data path, rules of data reading and loading, data augmentation parameters. To use it, a custom dataset config must be created for a certain dataset.
- Dataset: build a dataset class to index a data item, it can read a data item as 2D/3D form automatically based on the param "three_dim" in DatasetCfg.
- Dataloader: an data iteration, which take a Dataset instance and a DatasetCfg instance as its param. It can load a batch of 2D/3D data item automatically according to the param "three_dim" in DatasetCfg.
- DataAugmenter: automatically perform data augmentation for 2D/3D data.

How to use(details can be obtained from the code, it is really really simple):
Take [Synapse Abdomen dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) as an example,
- 1, build a dataset config class in DatasetCfg, define everything you need, such as data path, etc.
- 2, build a custom Dataset class inherited from Dataset, define the rules of reading a data item (def __getitem__(self, item))

The data is builded as a dict type through all steps:
- data item: {'data': ndarray, 'seg': ndarray, 'properties': dict, everything you want to keep}
- data batch: {'data': ndarray, 'seg': ndarray, 'indices': list, 'properties': dict, everything you want to keep},

# Installation
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)
- [medpy](https://github.com/loli/medpy)
