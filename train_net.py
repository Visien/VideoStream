import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn

import _init_paths  # noqa: F401

from pet.lib.utils.checkpointer import CheckPointer
from pet.lib.utils.comm import get_world_size, init_seed, is_main_process, synchronize
from pet.lib.utils.events import EventStorage
from pet.lib.utils.logger import build_train_hooks, write_metrics
from pet.lib.utils.lr_scheduler import LearningRateScheduler
from pet.lib.utils.misc import logging_rank, mismatch_params_filter, mkdir_p, setup_logging
from pet.lib.utils.optimizer import Optimizer

from pet.cnn.datasets.dataset import build_dataset, make_train_data_loader
from pet.cnn.modeling.model_builder import GeneralizedCNN
from pet.cnn.utils.analyser import RCNNAnalyser

from pet.projects.centerrit.core.config import get_base_cfg, infer_cfg


def train(cfg, model, loader, optimizer, scheduler, checkpointer, all_hooks):
    # switch to train mode
    model.train()

    # main loop
    start_iter = scheduler.iteration

    iteration = start_iter
    max_iter = len(loader)
    iter_loader = iter(loader)
    logging_rank("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter=start_iter, log_period=cfg.MISC.DISPLAY_ITER) as storage:
        try:
            for h in all_hooks:
                h.before_train()
            for iteration in range(start_iter, max_iter + 1):
                for h in all_hooks:
                    h.before_step(storage=storage)

                data_start = time.perf_counter()
                inputs, targets, _ = next(iter_loader)

                inputs = inputs.to(cfg.MISC.DEVICE)
                targets = [target.to(cfg.MISC.DEVICE) for target in targets]
                data_time = time.perf_counter() - data_start

                optimizer.zero_grad()

                outputs = model(inputs, targets)
                losses = sum(loss for loss in outputs["losses"].values())
                metrics_dict = outputs["losses"]
                metrics_dict["data_time"] = data_time
                write_metrics(metrics_dict, storage)
                losses.backward()

                # Due to weight decay,
                # if we don't manually set grad=None here,
                # weights in feature_adapt will decay to zero
                if (cfg.MODEL.AUXDET.STEP1_ENABLED
                    and iteration <= cfg.MODEL.AUXDET.DISTANCE_LOSS_WARMUP_ITERS):  # noqa: E129
                    for p in model.module.global_det.aux_det.feature_adapt.parameters():
                        p.grad = None

                optimizer.step()

                for h in all_hooks:
                    h.after_step(storage=storage)

                if is_main_process():
                    # Save model
                    if cfg.SOLVER.SNAPSHOT_ITER > 0 and iteration % cfg.SOLVER.SNAPSHOT_ITER == 0:
                        checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix="iter")
                storage.step()
        finally:
            if is_main_process() and iteration % cfg.SOLVER.SNAPSHOT_ITER != 0 and (iteration > 1000 or iteration == max_iter):
                checkpointer.save(model, optimizer, scheduler, copy_latest=True, infix="iter")

            for h in all_hooks:
                h.after_train(storage=storage)


def main(args):
    cfg = get_base_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg = infer_cfg(cfg, args.cfg_file)
    cfg.freeze()

    if not os.path.isdir(cfg.MISC.CKPT):
        mkdir_p(cfg.MISC.CKPT)
    setup_logging(os.path.join(cfg.MISC.CKPT, "log"), filename=f"train_{os.environ.get('CURRENT_TIME')}.txt")

    seed = init_seed(cfg.MISC.SEED)
    logging_rank(f"Training with seed: {seed}")

    if args.cfg_file is not None:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.MISC.CKPT, args.cfg_file.split("/")[-1]))
    for i in range(len(args.opts) // 2):
        logging_rank("Training with opts: {}  {}".format(args.opts[2 * i], args.opts[2 * i + 1]))

    # Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.ANALYSER.ENABLED:
        model = GeneralizedCNN(cfg)
        model.eval()
        analyser = RCNNAnalyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="flops")
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TRAIN.RESIZE.SCALES[0], cfg.TRAIN.RESIZE.SCALES[0], mode="activations")
        del model
    synchronize()

    # Create model
    model = GeneralizedCNN(cfg)
    logging_rank(model)
    logging_rank(
        "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | Activations: {:.4f}M / Conv_Activations: {:.4f}M"
        .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
    )

    # Create checkpointer
    checkpointer = CheckPointer(cfg.MISC.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME)

    # Load pre-trained weights or random initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MISC.CONV1_RGB2BGR)
    model.to(torch.device(cfg.MISC.DEVICE))
    if cfg.MISC.DEVICE == "cuda" and cfg.MISC.CUDNN:
        cudnn.benchmark = True
        cudnn.enabled = True

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank("The mismatch keys: {}".format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))))

    # Create training dataset and loader
    dataset = build_dataset(cfg, is_train=True)
    start_iter = checkpointer.checkpoint['scheduler']['iteration'] if checkpointer.resume else 1
    train_loader = make_train_data_loader(cfg, dataset, start_iter=start_iter)
    max_iter = len(train_loader)
    iter_per_epoch = max_iter // cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS

    # Some methods need to know present iter
    cfg.defrost()
    cfg.SOLVER.START_ITER = start_iter
    cfg.SOLVER.SCHEDULER.TOTAL_ITERS = max_iter
    cfg.freeze()

    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
    scheduler = checkpointer.load_scheduler(scheduler)

    # Precise BN
    precise_bn_args = [
        make_train_data_loader(cfg, dataset, start_iter=start_iter), model,
        torch.device(cfg.MISC.DEVICE)
    ] if cfg.TEST.PRECISE_BN.ENABLED else None

    # Model Distributed
    distributed = get_world_size() > 1
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Build hooks
    if cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS < 0:
        warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_ITERS
    else:
        warmup_iter = cfg.SOLVER.SCHEDULER.WARM_UP_EPOCHS * iter_per_epoch
    all_hooks = build_train_hooks(
        cfg, optimizer, scheduler, max_iter, warmup_iter, ignore_warmup_time=False, precise_bn_args=precise_bn_args
    )

    # Train
    train(cfg, model, train_loader, optimizer, scheduler, checkpointer, all_hooks)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pet Model Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cfg",
                        type=str,
                        dest="cfg_file",
                        default="cfgs/projects/centerrit/centerrit_DLA-34_adam_aug.yaml",
                        help="optional config file")
    parser.add_argument("opts",
                        nargs=argparse.REMAINDER,
                        help="See pet/projects/centerrit/core/config.py for all options")

    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    main(args)
