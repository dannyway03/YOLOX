#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
from pathlib import Path

import onnx
import onnxoptimizer
import onnxsim
import torch
from loguru import logger
from torch import nn

from yolox.exp import get_exp
from yolox.models.network_blocks import SiLU
from yolox.utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument("--weights",
                        default=None,
                        type=str,
                        help="path to pth model weights")
    parser.add_argument("-f","--exp_file",
                        default=None,
                        type=str,
                        help="experiment description file",
    )
    parser.add_argument("-t", "--type",
                        type=str,
                        default=None,
                        help="model type, yolox-s,m,l,x,nano,tiny")

    parser.add_argument('--img-size', nargs='+',
                        type=int,
                        default=[],
                        help='image size as H W')
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="batch size")
    parser.add_argument("--dynamic",
                        action="store_true",
                        help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--decode_in_inference",
                        action="store_true",
                        help="decode in inference or not"
    )
    parser.add_argument("--onnx-filename",
                        type=str,
                        default=None,
                        help="output name of onnx model"
    )
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp = get_exp(args.exp_file, args.type)
    exp.merge(args.opts)

    model = exp.get_model()

    logger.info('Loading model weights: {}'.format(args.weights))
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
        model.eval()
        model.load_state_dict(checkpoint['model'])
    else:
        logger.info('Loading checkpoint failed')
        exit(1)

    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode_in_inference

    if len(args.img_size) == 0:
        # override input image size
        args.img_size = [exp.test_size[0],exp.test_size[1]]

    dummy_input = torch.randn(args.batch_size, 3, args.img_size[0], args.img_size[1])

    if args.onnx_filename is None:
        args.onnx_filename = Path(args.weights).stem + '_' + str(args.batch_size) + 'x3x' + str(
                args.img_size[0]) + 'x' + str(args.img_size[1]) + '.onnx'

    f = '/'.join(['./weights/onnx', args.onnx_filename])
    torch.onnx.export(
            model,
            (dummy_input,),
            f,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch'},
                          'output': {0: 'batch'}} if args.dynamic else None,
            opset_version=13,
    )

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model

    try:
        logger.info('Starting to simplify ONNX...')
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, 'assert check failed'

        model_onnx = onnxoptimizer.optimize(onnx_model)
        onnx.save(onnx_model, f)

    except Exception as e:
        logger.info(f'Simplifier failure: {e}')

    logger.info('ONNX export success, saved as %s' % f)


if __name__ == "__main__":
    main()
