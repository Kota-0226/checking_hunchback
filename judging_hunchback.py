import argparse
import logging
import time
import sys

import cv2
import numpy as np

import os
from time import sleep

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
#from tf_pose import common

"""
#def save_frame_camera_key(device_num, dir_path, basename, n, ext='jpg', delay=1, window_name='frame'):
#cap = cv2.VideoCapture(device_num)

    #if not cap.isOpened():
        #return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        # if key == ord('c'):
        cv2.imwrite('{}_{}.{}'.format(
            base_path, n, ext), frame)

        if key == ord('q'):
            break

"""


# def save_frame_camera_key(dir_path, basename, n, image_name, ext='jpg', delay=1, window_name='frame'):
def save_frame_camera_key(dir_path, basename, n, image_name, ext='jpg'):
    #cap = cv2.VideoCapture(device_num)

    # if not cap.isOpened():
    # return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    """
    while True:  # このループから抜け出せてないのが、ダメなんじゃない？
        
        #ret, frame = cap.read()
        cv2.imshow(window_name, image_name)
        key = cv2.waitKey(delay) & 0xFF
        # if key == ord('c'):
    """
    cv2.imwrite('{}_{}.{}'.format(
        base_path, n, ext), image_name)

    if key == ord('q'):
        pass

    # print("Hello")

   # return 0

    # cv2.destroyWindow(window_name)


def findPoint(humans, p):  # 人間の姿勢を見つける関数
    for human in humans:
        try:
            body_part = human.body_parts[p]
            parts = [0, 0]

            # 座標を整数に切り上げで置換
            #parts[0] = int(body_part.x * width + 0.5)
            #parts[1] = int(body_part.y * height + 0.5)
            # parts = [x座標, y座標]

            parts[0] = int(body_part.x * w + 0.5)
            parts[1] = int(body_part.y * h + 0.5)
            return parts
        except:
            print("Not found")

        print("I am called!")


#save_frame_camera_key(0, 'data/temp', 'camera_capture')

# if __name__ == '__main__':
frame_id = 0

# print("Hello 1") これは呼ばれている

cap = cv2.VideoCapture(0)
ret_val, image = cap.read()
cv2.imshow("Camera", image)

while(1):
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    frame_id_str = str(frame_id)
    #print("Hello 2")
    save_frame_camera_key('data/temp', 'camera_capture',
                          frame_id_str, image)  # 写真は撮れた！

    #print("Hello 3")

    parser = argparse.ArgumentParser(description='tf-pose-estimation run')

    parser.add_argument(
        '--image', type=str, default='./data/temp/camaera_capture'+frame_id_str+'.jpg')  # 撮った写真を引数に追加
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. '
                        'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)

    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model),
                            target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    #image = common.read_imgfile(args.image, None, None)

    #height, width = image.shape[0], image.shape[1]

   # print("Hi")

    if image is None:
        #logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    t = time.time()
    humans = e.inference(image, resize_to_default=(
        w > 0 and h > 0), upsample_size=args.resize_out_ratio)

    tmp_nose = findPoint(humans, 0)
    nose = tmp_nose

    tmp_heart = findPoint(humans, 1)
    heart = tmp_heart

    right_shoulder = findPoint(humans, 2)
    left_shoulder = findPoint(humans, 5)

    print("nose={},{}".format(nose[0], nose[1]))  # printは毎回できる。
    print("heart={},{}".format(heart[0], heart[1]))

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if frame_id == 0:
        pass

    else:
        pass  # ここに後で、猫背かどうかを判定するチェックをする。（それか、elseは消して、これは最後にやるべきな気がしてきた。

    frame_id += 1
    time.sleep(3)  # 30s待つ
