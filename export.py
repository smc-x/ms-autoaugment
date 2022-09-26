"""Export checkpoint file into air, onnx, mindir models."""

import argparse

import numpy as np
from mindspore import (
    context,
    export,
    load_checkpoint,
    load_param_into_net,
    Tensor,
)

from src.config import Config
from src.network import WRN


parser = argparse.ArgumentParser(description='WRN with AutoAugment export.')
parser.add_argument(
    '--device_id', type=int, default=0,
    help='Device id.',
)
parser.add_argument(
    '--checkpoint_path', type=str, required=True,
    help='Checkpoint file path.',
)
parser.add_argument(
    '--file_name', type=str, default='wrn-autoaugment',
    help='Output file name.',
)
parser.add_argument(
    '--file_format', type=str, choices=['AIR', 'ONNX', 'MINDIR'],
    default='MINDIR', help='Export format.',
)
parser.add_argument(
    '--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'],
    default='Ascend', help='Device target.',
)


if __name__ == '__main__':
    args = parser.parse_args()
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target=args.device_target,
    )
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)

    conf = Config(training=False, load_args=False)
    net = WRN(160, 3, conf.class_num)

    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, param_dict)

    image = Tensor(np.ones((1, 3, 32, 32), np.float32))
    export(
        net, image,
        file_name=args.file_name,
        file_format=args.file_format,
    )
