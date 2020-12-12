# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import numpy as np
import torch

from .utils import get_affine_transform
from .utils import ctdet_decode
from .utils import ctdet_post_process
from .pedestrian_detector import PedestrianDetector
from .network.resnet import get_resnet


class PedestrianDetectorCenterNetParammeters(object):

    def __init__(self):

        self.model_arch = "res_18"

        self.class_num = 2

        self.max_object_num = 100

        self.head_conv_channels = 64

        self.down_ratio = 4

        self.input_h = 512
        self.input_w = 512

        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio

        self.input_resolution = max(self.input_h, self.input_w)
        self.output_resolution = max(self.output_h, self.output_w)

        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]

        self.heads = {
            "hm": self.class_num,
            "wh": 2,
            "reg": 2
        }

        return


class PedestrianDetectorCenterNet(PedestrianDetector):


    def __init__(self, model_path, gpu_ids=[], score_threshold=0.4):

        super(PedestrianDetectorCenterNet, self).__init__()
        self.parameters = PedestrianDetectorCenterNetParammeters()
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.score_threshold = score_threshold

        if len(self.gpu_ids) != 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = \
                ",".join(list(map(str, self.gpu_ids)))
            self.parameters.device = torch.device("cuda")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.parameters.device = torch.device("cpu")

        self.model = self.__create_model(
            self.parameters.model_arch, self.parameters.heads,
            self.parameters.head_conv_channels)
        self.model = self.__load_model(self.model, self.model_path)
        self.model = self.model.to(self.parameters.device)
        self.model.eval()

        self.mean = torch.Tensor(self.parameters.mean) \
            .to(self.parameters.device)
        self.std = torch.Tensor(self.parameters.std) \
            .to(self.parameters.device)
        self.max_per_image = self.parameters.max_object_num
        self.num_classes = self.parameters.class_num
        self.scale = 1
        self.pause = True

    def detect(self, detect_image, min_bbox_size=None):

        ret = {"pedestrian": None, "head": None}

        if detect_image is None:
            return ret
        img_h, img_w = detect_image.shape[0], detect_image.shape[1]

        images, affine_info = self.__pre_process(detect_image)
        images = images.to(self.parameters.device)
        images = (images / 255. - self.mean) / self.std  # 图片归一化
        images = images.permute(2, 0, 1).reshape(
            1, 3, self.parameters.input_h, self.parameters.input_w)
        torch.cuda.synchronize()

        network_results = self.__process(images)
        torch.cuda.synchronize()

        detect_results = [self.__post_process(network_results,
                                              affine_info, self.scale)]
        torch.cuda.synchronize()

        detect_results = self.__merge_outputs(detect_results)
        torch.cuda.synchronize()

        for detect_class in detect_results.keys():
            result = detect_results[detect_class]

            if min_bbox_size is not None:
                index = np.where(
                    (result[:, -1] > self.score_threshold) &
                    ((result[:, 2] - result[:, 0]) > min_bbox_size) &
                    ((result[:, 3] - result[:, 1]) > min_bbox_size))[0]
            else:
                index = np.where((result[:, -1] > self.score_threshold))[0]

            result[index, :-1] = np.maximum(result[index, :-1], 0)
            result[index, 0] = np.minimum(result[index, 0], img_w)
            result[index, 1] = np.minimum(result[index, 1], img_h)
            result[index, 2] = np.minimum(result[index, 2], img_w)
            result[index, 3] = np.minimum(result[index, 3], img_h)
            if detect_class == 1:
                ret["pedestrian"] = result[index, :]
            elif detect_class == 2:
                ret["head"] = result[index, :]
            else:
                raise Exception("Detect result appear wrong class, please check!")
        return ret

    def __create_model(self, model_arch, heads, head_conv_channels):

        layer_num = int(model_arch[model_arch.find("_") + 1:])
        model = get_resnet(layer_num=layer_num, heads=heads,
                           head_conv_channels=head_conv_channels)
        return model

    def __load_model(self, model, model_path):

        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        checkpoint_state_dict = checkpoint["state_dict"]
        state_dict = {}

        for k in checkpoint_state_dict:
            if k.startswith("module") and not k.startswith("module_list"):
                state_dict[k[7:]] = checkpoint_state_dict[k]
            else:
                state_dict[k] = checkpoint_state_dict[k]
        model_state_dict = model.state_dict()

        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print("Skip loading parameter {}.".format(k))
                    print("Required {}, ".format(model_state_dict[k].shape),
                          "loaded {}.".format(state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print("Drop parameter {}.".format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print("No param {}.".format(k))
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        print("Successfully Load CenterNet Model")
        return model

    def __pre_process(self, image):

        height, width = image.shape[0:2]
        inp_height = self.parameters.input_h
        inp_width = self.parameters.input_w
        center = np.array([width / 2., height / 2.], dtype=np.float32)
        scale = max(height, width) * 1.0
        trans_input_matrix = get_affine_transform(
            center, scale, 0, [inp_width, inp_height])  # 仿射矩阵

        inp_image = cv2.warpAffine(
            image, trans_input_matrix, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR
        )
        images = torch.from_numpy(inp_image.astype(np.float32))
        affine_info = {
            "center": center,
            "scale": scale,
            "out_height": inp_height // self.parameters.down_ratio,
            "out_width": inp_width // self.parameters.down_ratio
        }
        return images, affine_info

    def __process(self, images):

        with torch.no_grad():
            output = self.model(images)[-1]
            heatmap = output["hm"].sigmoid_()
            width_height = output["wh"]
            regress_offset = output["reg"]
            torch.cuda.synchronize()
            network_results = ctdet_decode(
                heatmap, width_height, reg=regress_offset,
                k=self.parameters.max_object_num
            )
        return network_results

    def __post_process(self, results, affine_info, scale=1):

        results = results.detach().cpu().numpy()
        results = results.reshape(1, -1, results.shape[2])
        results = ctdet_post_process(
            results.copy(), [affine_info["center"]],
            [affine_info["scale"]], affine_info["out_height"],
            affine_info["out_width"], self.num_classes)
        for j in range(1, self.num_classes + 1):
            results[0][j] = np.array(results[0][j],
                                     dtype=np.float32).reshape(-1, 5)
            results[0][j][:, :4] /= scale
        return results[0]

    def __merge_outputs(self, detect_results):

        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detect_results], axis=0
            ).astype(np.float32)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)]
        )
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results
