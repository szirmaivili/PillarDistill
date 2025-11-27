import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time 
from det3d.torchie.apis.train import example_to_device
#import tensorflow as tf

# Ez az, ami új:

from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss, IouLoss, IouRegLoss
from DetectionNet import DetectionNet_mod
from backbone_dense import Backbone_mod
from neck import Neck_mod
from bbox_heads import Bbox
from types import SimpleNamespace
from det3d.models.readers.dynamic_pillar_encoder import DynamicPillarFeatureNet
from tqdm import tqdm
from quantization import make_quantization_aware_pytorch as quant

tasks = [
    dict(stride=8, class_names=["car"]),
    dict(stride=8, class_names=["truck", "construction_vehicle"]),
    dict(stride=8, class_names=["bus", "trailer"]),
    dict(stride=8, class_names=["barrier"]),
    dict(stride=8, class_names=["motorcycle", "bicycle"]),
    dict(stride=8, class_names=["pedestrian", "traffic_cone"]),
    ]

target_assigner = dict(
    tasks=tasks,
)

pillar_size=0.075
pc_range=[-54, -54, -5.0, 54, 54, 3.0]

DOUBLE_FLIP = False

assigner = SimpleNamespace(
    target_assigner=target_assigner,
    out_size_factor=8,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    pc_range=pc_range,
    voxel_size=[pillar_size, pillar_size, 0.2],
)

train_cfg = SimpleNamespace(assigner=assigner)

test_cfg = SimpleNamespace(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=SimpleNamespace(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=[1000]*6,
        nms_post_max_size=[83]*6,
        nms_iou_threshold=[0.2]*6,
    ),
    score_threshold=0.1,
    rectifier=[0.5]*6,
    pc_range=pc_range[:2],
    out_size_factor=8,
    voxel_size=[pillar_size, pillar_size],
    double_flip=DOUBLE_FLIP,
    return_raw=False,
    per_class_nms = False,
    circular_nms = False
)

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--preds", type=str, help="Prediction pickle location")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    # model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    code_weights = [1.0,1.0,1.0,1.0,1.0,1.0,0.2,0.2,1.0,1.0]
    task_strides = [t["stride"] for t in tasks]

    backbone = quant(Backbone_mod())
    neck = quant(Neck_mod())
    bbox = quant(Bbox())
    reader = DynamicPillarFeatureNet(num_input_features = 5, num_filters=(32,), pillar_size=pillar_size, pc_range=pc_range)

    model = DetectionNet_mod(
        backbone=backbone,
        neck=neck,
        bbox_head=bbox,
        reader=reader,
        crit=FastFocalLoss(),
        crit_reg=RegLoss(),
        crit_iou=IouLoss(),
        crit_iou_reg=IouRegLoss("DIoU"),
        code_weights=code_weights,
        weight=0.25,
        with_iou=True,
        with_iou_reg=True,
        task_strides=task_strides,
        dataset="nuscenes",
        train_cfg=train_cfg,
        test_cfg=test_cfg
    )

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output,}
            )
            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)

    print("\n Total time per frame: ", (time_end -  time_start) / (end - start))

    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # with open(r'/PillarDistill/real_bev_distill_test_new/predictions.pkl', 'rb') as f:

    #     predictions = pickle.load(f)

    # def tf_to_torch(x):
    #     if isinstance(x, tf.Tensor):
    #         return torch.from_numpy(x.numpy())
    #     elif isinstance(x, dict):
    #         return {k: tf_to_torch(v) for k, v in x.items()}
    #     elif isinstance(x, list):
    #         return [tf_to_torch(v) for v in x]
    #     else:
    #         return x
        
    # predictions = {k: tf_to_torch(v) for k,v in predictions.items()}

    # with open(args.preds, 'rb') as f:

    #     predictions = pickle.load(f)

    for k, v in predictions.items():

        if v.get('metadata') is None:
            v['metadata'] = {}
        v['metadata']['token'] = k

    save_pred(predictions, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=args.work_dir, testset=args.testset)

    # if result_dict is not None:
    #     for k, v in result_dict["results"].items():
    #         print(f"Evaluation {k}: {v}")

    # if args.txt_result:
    #     assert False, "No longer support kitti"

    #print(model)

if __name__ == "__main__":
    main()
