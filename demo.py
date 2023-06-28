# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import gc
import glob
import multiprocessing as mp
import os
import pdb
import time
import cv2
import torch.cuda
import tqdm
import imageio

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo

# constants
WINDOW_NAME = "Open vocabulary segmentation"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for open vocabulary segmentation")
    parser.add_argument(
        "--config-file",
        default="configs/ovseg_swinB_vitL_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="A list of user-defined class_names"
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    class_names = args.class_names
    if args.input:
        print(args.input)
        print(args.output)
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        inputs = []
        if os.path.isdir(args.input[0]):
            root_dir = args.input[0]
            for frame_index in os.listdir(root_dir):
                frame_dir = os.path.join(root_dir,frame_index)
                for frame_name in os.listdir(frame_dir):
                    frame_name_dir = os.path.join(frame_dir, frame_name)
                    for sensor_name in os.listdir(frame_name_dir):
                        sensor_dir = os.path.join(frame_name_dir, sensor_name)
                        if "camera" in sensor_name:
                            inputs.append(os.path.join(sensor_dir, os.listdir(sensor_dir)[0]))
        else:
            for path in args.input:
                inputs.append(path)
        #print(inputs)
        for path in tqdm.tqdm(inputs, disable=not args.output):
            # use PIL, to be consistent with evaluation
            #pdb.set_trace()
            #if "upmiddle_middle" not in path:
            #    continue
            img = imageio.v3.imread(path)
            #img_shape = (int(img.shape[1]*4/5), int(img.shape[0]*4/5))

            img_shape = (img.shape[1], img.shape[0])
            print(img_shape)
            if img_shape[0] >= 2304*0.9:
                img_shape = (int(2304*0.9), int(1728*0.9))
            print(f"res:{img_shape}, path:{path.split('/')[-1]}")
            if img.shape[1] != img_shape[0]:
                img = cv2.resize(img, img_shape)
            new_path = path.split("/")[-1]
            new_path = os.path.join(args.output, os.path.basename(path)).replace(".jpg", f"_{img_shape[1]}_{img_shape[0]}.jpg")

            #new_path = path.replace(".jpg", f"_{img_shape[1]}_{img_shape[0]}.jpg")
            imageio.imwrite(new_path, img)
            img = read_image(new_path, format="BGR")

            start_time = time.time()
            predictions, visualized_output, image_pixel_feature = demo.run_on_image(img, class_names)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
            gc.collect()
            torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
            #time.sleep(5)


    else:
        raise NotImplementedError