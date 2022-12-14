"""postprocess for 310 inference"""
import os
import argparse
import numpy as np
from mindspore.nn import Top1CategoricalAccuracy
parser = argparse.ArgumentParser(description="postprocess")
label_path = "./preprocess_Result/cifar10_label_ids.npy"
parser.add_argument("--result_dir", type=str, default="./result_Files", help="result files path.")
parser.add_argument('--dataset_name', type=str, default="cifar10")
parser.add_argument("--label_dir", type=str, default=label_path, help="image file path.")
args = parser.parse_args()

def calcul_acc(lab, preds):
    return sum(1 for x, y in zip(lab, preds) if x == y) / len(lab)
if __name__ == '__main__':
    if args.dataset_name == "cifar10":
        top1_acc = Top1CategoricalAccuracy()
        rst_path = args.result_dir
        labels = np.load(args.label_dir, allow_pickle=True)
        for idx, label in enumerate(labels):
            f_name = os.path.join(rst_path, "autoaugment_data"  + "_" + str(idx) + "_0.bin")
            pred = np.fromfile(f_name, np.float32)
            pred = pred.reshape(1, int(pred.shape[0] / 1))
            top1_acc.update(pred, labels[idx])
        print("acc: ", top1_acc.eval())
