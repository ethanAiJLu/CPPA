import os.path as osp
import pickle
from dassl.utils import listdir_nohidden, mkdir_if_missing
from dassl.data.datasets.build import DATASET_REGISTRY
from dassl.data.datasets.base_dataset import Datum, DatasetBase
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
import os

@DATASET_REGISTRY.register()
class miniDomainNet_CPPA(DatasetBase):
    """A subset of DomainNet.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
        - Zhou et al. Domain Adaptive Ensemble Learning.
    """

    dataset_dir = "Mini_DomainNet"
    domains = ["clipart_train", "clipart_test", "painting_train", "painting_test", "real_train", "real_test", "sketch_train", "sketch_test"]
    
    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        print(self.dataset_dir)
        print(cfg.DATASET.SOURCE_DOMAINS[0])
        print(cfg.DATASET.TARGET_DOMAINS[0])
        self.image_dir = os.path.join(self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS[0]+"_train")
        self.split_path = os.path.join(self.dataset_dir, f"split_{cfg.DATASET.SOURCE_DOMAINS[0]}_train.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, f"{cfg.DATASET.SOURCE_DOMAINS[0]}_train_split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, _, _ = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        train_x = train
        u_name = cfg.DATASET.TARGET_DOMAINS[0] + "_train"
        t_name = cfg.DATASET.TARGET_DOMAINS[0] + "_test"
        train_u = self._read_data([u_name])
        test = self._read_data([t_name])

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