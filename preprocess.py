"""preprocess"""
import os
import numpy as np
from src.config import Config
from src.dataset import create_cifar10_dataset
from src.utils import init_utils

if __name__ == "__main__":
    conf = Config(training=False)
    init_utils(conf)
    if conf.dataset == "cifar10":
        dataset = create_cifar10_dataset(
            dataset_path=conf.dataset_path,
            do_train=False,
            repeat_num=1,
            batch_size=1,
            target=conf.device_target,
        )
        img_path = os.path.join('./preprocess_Result/', "00_data")
        os.makedirs(img_path)
        label_list = []
        for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            file_name = "autoaugment_data" + "_" + str(idx) + ".bin"
            file_path = os.path.join(img_path, file_name)
            data["image"].tofile(file_path)
            label_list.append(data["label"])
        np.save(os.path.join('./preprocess_Result/', "cifar10_label_ids.npy"), label_list)
        print("=" * 20, "export bin files finished", "=" * 20)
