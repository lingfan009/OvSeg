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

def get_image_embedding(config_file, scene_id, cam, class_names, input_image, model_weights):
    mp.set_start_method("spawn", force=True)

    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(model_weights.split(" "))
    cfg.freeze()


    if cam == "camera_upmiddle_left":
        torch.cuda.set_device(1)
    else:
        torch.cuda.set_device(2)
    demo = VisualizationDemo(cfg, cam)
    img = imageio.v3.imread(input_image) # R G B
    predictions, visualized_output, image_pixel_feature, text_features = demo.run_on_image(img, cam, class_names)
    output_dir = os.path.join("/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/", "image_with_label", scene_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.join(output_dir, os.path.basename(input_image))
    visualized_output.save(out_filename)

    img_save_dir = os.path.join("/home/fan.ling/big_model/OpenScene/OpenScene/fuse_2d_features/nuscenes_autra_2d_test/", "image_without_label", scene_id)
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)
    img_save_filename = os.path.join(img_save_dir, os.path.basename(input_image))
    imageio.v3.imwrite(img_save_filename, img)

    gc.collect()
    torch.cuda.empty_cache()
    return image_pixel_feature, text_features

def get_images_embedding(config_file, class_names, input_images, model_weights):
    mp.set_start_method("spawn", force=True)

    # load config from file and command-line arguments
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(model_weights.split(" "))
    cfg.freeze()

    demo = VisualizationDemo(cfg)

    pixel_embedings_results = []
    # print(inputs)
    for path in tqdm.tqdm(input_images, disable=False):
        img = imageio.v3.imread(path)

        img_shape = (img.shape[1], img.shape[0])
        print(img_shape)
        # if img_shape[0] >= 2304 * 0.9:
        #     img_shape = (int(2304 * 0.1), int(1728 * 0.1))
        print(f"res:{img_shape}, path:{path.split('/')[-1]}")
        if img.shape[1] != img_shape[0]:
            img = cv2.resize(img, img_shape)
        print(img.shape)
        predictions, visualized_output, image_pixel_feature = demo.run_on_image(img, class_names)
        output_dir = "output/autra_test_data/"

        out_filename = os.path.join(output_dir, os.path.basename(path))

        visualized_output.save(out_filename)

        gc.collect()
        torch.cuda.empty_cache()
        pixel_embedings_results.append(image_pixel_feature)
    return pixel_embedings_results



if __name__ == "__main__":

    if True:
        print("enter")
        config_file = "/home/fan.ling/big_model/OvSeg/OvSeg/configs/ovseg_swinB_vitL_demo.yaml"
        class_names = 'car,tree,grass,pole,road,cyclist,vehicle,truck,bicycle,other flat,buildings,safety barriers,sidewalk,manmade,sky,bus,suv,person,rider'
        class_names = class_names.split(',')
        model_weights = 'MODEL.WEIGHTS /home/fan.ling/big_model/OvSeg/OvSeg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth'

        inputs = []
        root_dir = "resources/autra_test_data/"
        root_dir = "/home/fan.ling/big_model/OpenScene/OpenScene/data/nuscenes_autra_2d_test/train/1684326201022-Robin/color/"
        for image_name in os.listdir(root_dir):
            input_file = os.path.join(root_dir, image_name)
            feat_2ds = get_image_embedding(config_file, class_names, input_file, model_weights)
            print(len(feat_2ds))
            print(feat_2ds.shape)
    else:
        mp.set_start_method("spawn", force=True)
        args = get_parser().parse_args()
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        demo = VisualizationDemo(cfg)
        class_names = args.class_names
        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            inputs = []
            # if os.path.isdir(args.input[0]):
            #     root_dir = args.input[0]
            #     for frame_index in os.listdir(root_dir):
            #         frame_dir = os.path.join(root_dir,frame_index)
            #         for frame_name in os.listdir(frame_dir):
            #             frame_name_dir = os.path.join(frame_dir, frame_name)
            #             for sensor_name in os.listdir(frame_name_dir):
            #                 sensor_dir = os.path.join(frame_name_dir, sensor_name)
            #                 if "camera" in sensor_name:
            #                     inputs.append(os.path.join(sensor_dir, os.listdir(sensor_dir)[0]))
            if os.path.isdir(args.input[0]):
                root_dir = args.input[0]
                for image_name in os.listdir(root_dir):
                    inputs.append(os.path.join(root_dir, image_name))
            else:
                for path in args.input:
                    inputs.append(path)
            print(inputs)
            for path in tqdm.tqdm(inputs, disable=not args.output):
                # use PIL, to be consistent with evaluation
                
                img = imageio.v3.imread(path)            
                img_shape = (int(img.shape[1]*0.6), int(img.shape[0]*0.6))
                print(img_shape)
                img = cv2.resize(img, img_shape)
                print(img.shape)

                new_path = os.path.join(args.output, os.path.basename(path)).replace(".jpg", f"_{img_shape[0]}_{img_shape[1]}.jpg")
                imageio.imwrite(new_path, img)
                #new_path = path
                
                img = read_image(new_path, format="BGR")
                print(img.shape)

                start_time = time.time()
                predictions, visualized_output, image_pixel_feature = demo.run_on_image(img, class_names)
                print(image_pixel_feature.shape)
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



        else:
            raise NotImplementedError