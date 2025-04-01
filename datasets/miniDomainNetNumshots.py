import os.path as osp

from dassl.utils import listdir_nohidden
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase
import random

@DATASET_REGISTRY.register()
class miniDomainNetNumshots(DatasetBase):
    """A subset of DomainNet.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning.
    """

    dataset_dir = "domainnet"
    # dataset_dir = "DomainNet"
    domains = ["clipart", "painting", "real", "sketch"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_dir = osp.join(self.dataset_dir, "splits_mini")

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="train")
        train_u = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="train")
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, split="test")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, split="test")
        if cfg.DATASET.NUM_SHOTS > 0:
            train_x = self.sample_num_shots(train_x,cfg.DATASET.NUM_SHOTS)
        self.target_portion = self.portion(train_u)

        super().__init__(train_x=train_x, train_u=train_u, val=val, test=test)
        
        
    def sample_num_shots(self,items,num_shots):
        label_2_samples = {}
        for item in items:
            item_label = item.label
            if item_label in label_2_samples:
                label_2_samples[item_label].append(item)
            else:
                label_2_samples[item_label] = [item]
        targets = []
        for label, items in label_2_samples.items():
            item_target = []
            if num_shots > len(items):
                n = int(num_shots/len(items))
                for i in range(n):
                    item = random.sample(items, len(items))
                    item_target.extend(item)
                item = random.sample(items,num_shots - len(item_target))
                item_target.extend(item)
            else:
                item = random.sample(items, num_shots)
                item_target.extend(item)
            targets.extend(item_target)
        return targets
    
    def portion(self,items):
        sum = len(items)
        item_dict = {}
        for item in items:
            item_label = item.label
            if item_label in item_dict:
                item_dict[item_label]+=1
            else:
                item_dict[item_label] = 1
        por = [item_dict[key]/sum for key in sorted(item_dict.keys())]
        return por
    

    def _read_data(self, input_domains, split="train"):
        items = []

        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)

        return items
