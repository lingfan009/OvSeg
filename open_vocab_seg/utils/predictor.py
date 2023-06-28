# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import pdb

import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


# fmt: off
# RGB:
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on

class OVSegPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # height = int(height / 2)
            # width = int(width / 2)
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, "class_names": class_names}
            predictions = self.model([inputs])[0]
            return predictions

class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, mask_proposal_embed, cls_index_map, text_features, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        device = mask_proposal_embed.device
        clip_adapter_embed_tmp = mask_proposal_embed[:,:5]
        clip_adapter_embed = mask_proposal_embed
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        print(f"sem_seg : labels:{labels}, areas:{areas}")
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        #image_pixel_feature = np.zeros((sem_seg.shape[0], sem_seg.shape[1], 768))
        image_pixel_feature = torch.zeros(sem_seg.shape[0], sem_seg.shape[1], 768, device=device)


        mask_type = torch.zeros(len(labels), 768, device=device)
        index = 0
        for label in filter(lambda l: l < len(class_names), labels):
            binary_mask = (sem_seg == label).astype(np.uint8)
            image_pixel_feature[binary_mask] = clip_adapter_embed[cls_index_map[label]]
            mask_type[index] = clip_adapter_embed[cls_index_map[label]]
            index += 1
        #pdb.set_trace()
        tmp = (mask_type @ text_features.T)[:,:-1]
        mask_type_result = torch.argmax((mask_type @ text_features.T)[:,:-1], dim=1)

        image_label_result = torch.argmax((image_pixel_feature @ text_features.T)[:,:,:-1], dim=2)

        labels, areas = np.unique(image_label_result.detach().cpu().numpy(), return_counts=True)
        print(f"image_label_result : labels:{labels}, areas:{areas}")
        sem_seg = image_label_result.detach().cpu().numpy()
        print(class_names)
        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            # set color
            mask_color = _COLORS[label]
            binary_mask = (sem_seg == label).astype(np.uint8)
            #image_pixel_feature[binary_mask] = clip_adapter_embed[cls_index_map[label]]
            text = class_names[label]
            print(f"text:{text}, mask_color:{mask_color}")
            #if text == "road":
            #    continue
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                #edge_color=(1.0, 1.0, 240.0 / 255),
                edge_color=(0.0,  0.70, 0.78),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        image_pixel_feature = None

        return self.output, image_pixel_feature



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        print(cfg)
        self.clip_ensemble_weight = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = OVSegPredictor(cfg)

    def run_on_image(self, image, class_names):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image, class_names)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
        if "sem_seg" in predictions:
            r = predictions["sem_seg"]
            mask_former_embed = predictions["ori_mask_former_embed"]
            clip_adapter_embed = predictions["ori_clip_adapter_embed"]
            cls_index_map = predictions["cls_index_map"]
            text_features = predictions["text_features"]
            if self.clip_ensemble_weight > 0:
                mask_proposal_embed = torch.pow(mask_former_embed, 1 - self.clip_ensemble_weight) * \
                               torch.pow(clip_adapter_embed, self.clip_ensemble_weight)
            else:
                print("only use clip_adapter_embed")
                mask_proposal_embed = clip_adapter_embed
            blank_area = (r[0] == 0)
            pred_mask = r.argmax(dim=0).to('cpu')
            pred_mask[blank_area] = 255
            pred_mask = np.array(pred_mask, dtype=np.int)


            vis_output, image_pixel_feature = visualizer.draw_sem_seg(
                pred_mask,
                mask_proposal_embed,
                cls_index_map,
                text_features
            )
        else:
            raise NotImplementedError

        return predictions, vis_output, image_pixel_feature