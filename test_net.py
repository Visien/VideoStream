import argparse
import os

import torch

import _init_paths  # noqa: F401

from pet.lib.utils.checkpointer import get_weights, load_weights
from pet.lib.utils.comm import all_gather, init_seed, is_main_process, synchronize
from pet.lib.utils.logger import build_test_hooks
from pet.lib.utils.misc import logging_rank, mkdir_p, setup_logging
from pet.lib.utils.timer import Timer

from pet.cnn.core.test import TestEngine
from pet.cnn.datasets.dataset import build_dataset, make_test_data_loader
from pet.cnn.datasets.postprocess import CNNPostProcessor
from pet.cnn.modeling.model_builder import GeneralizedCNN
from pet.cnn.utils.analyser import RCNNAnalyser

from pet.projects.centerrit.core.config import get_base_cfg, infer_cfg


def test(cfg, test_engine, loader, dataset, all_hooks):
    total_timer = Timer()
    total_timer.tic()
    all_results = [[] for _ in range(7)]
    processor = CNNPostProcessor(cfg, dataset)
    with torch.no_grad():
        loader = iter(loader)
        for i in range(len(loader)):
            all_hooks.iter_tic()
            all_hooks.data_tic()
            inputs, targets, idx = next(loader)
            all_hooks.data_toc()
            all_hooks.infer_tic()

            result = test_engine(inputs, targets)
            print(inputs, targets)
            print(result)

            all_hooks.infer_toc()
            all_hooks.post_tic()

            eval_results = processor(inputs, result, idx, targets)
            all_results = [results + eva for results, eva in zip(all_results, eval_results)]

            all_hooks.post_toc()
            all_hooks.iter_toc()
            if is_main_process():
                all_hooks.log_stats(i, 0, len(loader), len(dataset))

    all_results = list(zip(*all_gather(all_results)))
    all_results = [[item for sublist in results for item in sublist] for results in all_results]
    if is_main_process():
        total_timer.toc(average=False)
        logging_rank("Total inference time: {:.3f}s".format(total_timer.average_time))
        processor.close(all_results)


def main(args):
    cfg = get_base_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg = infer_cfg(cfg)
    cfg.freeze()

    init_seed(cfg.MISC.SEED)

    if not os.path.isdir(cfg.MISC.CKPT):
        mkdir_p(cfg.MISC.CKPT)
    setup_logging(os.path.join(cfg.MISC.CKPT, "log"), filename=f"test_{os.environ.get('CURRENT_TIME')}.txt")

    for i in range(len(args.opts) // 2):
        logging_rank("Testing with opts: {}  {}".format(args.opts[2 * i], args.opts[2 * i + 1]))

    # Calculate Params & FLOPs & Activations
    n_params, conv_flops, model_flops, conv_activs, model_activs = 0, 0, 0, 0, 0
    if is_main_process() and cfg.ANALYSER.ENABLED:
        model = GeneralizedCNN(cfg)
        model.eval()
        analyser = RCNNAnalyser(cfg, model, param_details=False)
        n_params = analyser.get_params()[1]
        conv_flops, model_flops = analyser.get_flops_activs(cfg.TEST.RESIZE.SCALE, cfg.TEST.RESIZE.SCALE, mode="flops")
        conv_activs, model_activs = analyser.get_flops_activs(cfg.TEST.RESIZE.SCALE, cfg.TEST.RESIZE.SCALE,
                                                              mode="activations")
        del model
    synchronize()

    # Create model
    model = GeneralizedCNN(cfg)
    logging_rank(model)
    logging_rank(
        "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M"
        .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
    )

    # Load model
    test_weights = get_weights(cfg.MISC.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, test_weights)
    model.eval()
    model.to(torch.device(cfg.MISC.DEVICE))

    # Create testing dataset and loader
    dataset = build_dataset(cfg, is_train=False)
    test_loader = make_test_data_loader(cfg, dataset)

    # Build hooks
    all_hooks = build_test_hooks(args.cfg_file.split("/")[-1], log_period=10, num_warmup=0)

    # Build test engine
    test_engine = TestEngine(cfg, model)

    # Test
    test(cfg, test_engine, test_loader, dataset, all_hooks)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Pet Model Testing")
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
