import os.path as osp
import pickle
import torch
from dassl.utils import listdir_nohidden, mkdir_if_missing

from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase

@DATASET_REGISTRY.register()
class DARS_AID_UCM(DatasetBase):
    """
    Statistics:
        - AID have 4,510 images.
        - UCM have 1,300 images.
        - 13 classes related to remote sensoring objects.
    """
    dataset_dir = "aid_ucm"

    domains = ["AID", "UCM"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS)
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS)
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS)

        super().__init__(train_x=train_x, train_u=train_u, test=test)

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