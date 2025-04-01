import os.path as osp
import pickle

from dassl.utils import listdir_nohidden, mkdir_if_missing
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class DARS_UCM_NWPU_ONE(DatasetBase):
    """
    Statistics:
        - NWPU have 14,000 images.
        - UCM have 2,000 images.
        - 20 classes related to remote sensoring objects.
    """
    dataset_dir = "ucm_nwpu"

    domains = ["NWPU", "UCM"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        test = self._read_data(cfg.DATASET.SOURCE_DOMAINS)

        super().__init__(train_x=train_x, test=test)

    def _read_data(self, input_domains):
        items = []

        for domain, dname in enumerate(input_domains):
            domain_dir = osp.join(self.dataset_dir, dname)
            class_names = listdir_nohidden(domain_dir)
            class_names.sort()

            for label, class_name in enumerate(class_names):
                class_path = osp.join(domain_dir, class_name)
                imnames = listdir_nohidden(class_path)

                for imname in imnames:
                    impath = osp.join(class_path, imname)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=class_name.lower(),
                    )
                    items.append(item)

        return items