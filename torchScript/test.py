# -*- coding: utf-8 -*-

import cv2
import os
import shutil
import time
from detector_centernet.pedestrian_detector_centernet import PedestrianDetectorCenterNet

def pedestrian_detect(model_path, img_dir, gpu_ids, save_dir):

    detect_model = PedestrianDetectorCenterNet(
        model_path=model_path, gpu_ids=gpu_ids, score_threshold=0.4)
    N = len(os.listdir(img_dir))
    detect_time = 0
    print("--- Total Number of Images: {}".format(N))

    for img_name in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, img_name))

        start = time.time()
        detect_result = detect_model.detect(img)
        detect_time += (time.time() - start)
        draw_result(os.path.join(save_dir, img_name), img, detect_result)

    print("--- Consuming Time of CenterNet: {}".format(detect_time))
    print("--- Speed of CenterNet: [ {} ]fps\n".format(N / detect_time))


def draw_result(save_path, img, detect_result):

    color_dict = {"pedestrian": (255, 0, 0), "head": (0, 255, 0)}
    if detect_result is None:
        print("Image {} is None".format(detect_result.image_name))

    for category_name in detect_result.keys():
        category_detect_result_list = detect_result[category_name]
        for result in category_detect_result_list:
            x1, y1, x2, y2, score = result
            cv2.rectangle(img, (x1, y1), (x2, y2), color_dict[category_name], thickness=2)
            cv2.putText(img, str(category_name) + ' ' + str(round(score, 2)), (int(x1 + 2), int(y2 - 8)),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color_dict[category_name], thickness=2)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":

    model_path = r"./pedestrian_head_detect_v0.0.1_52374f84c19627bd505c411c771befa1.zip"
    gpu_ids = [3]
    img_dir = r"./images"
    save_dir = r"./result"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)


    detect_result_list = pedestrian_detect(model_path, img_dir, gpu_ids, save_dir)




