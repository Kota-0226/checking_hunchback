import argparse
import cv2

import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose import common

# https://cam-inc.co.jp/p/techblog/603142844403680193 参照

parser = argparse.ArgumentParser(description='姿勢推定する')
parser.add_argument('--camera', type=str, default=0)
parser.add_argument('--resize', type=str, default='432x368')
parser.add_argument('--resize-out-ratio', type=float, default=4.0)
parser.add_argument('--model', type=str, default='mobilenet_thin')
args = parser.parse_args()
w, h = model_wh(args.resize)

"""
def findPoint(humans, p):
    for human in humans:
        try:
            body_part = human.body_parts[p]
            parts = [0, 0]

            # 座標を整数に切り上げで置換
            parts[0] = int(body_part.x * width + 0.5)
            parts[1] = int(body_part.y * height + 0.5)
            # parts = [x座標, y座標]
            return parts
        except:
            print("Not found")
"""


cam = cv2.VideoCapture(args.camera)

while True:
    ret_val, image = cam.read()
    cv2.imshow("Camera", image)
# ret_val, image = cam.read()  # 後々これを、写真にすればいいのではなかろうか

# 検知された人間

# 高さと幅を取得
    height, width = image.shape[0], image.shape[1]

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    humans = e.inference(image, resize_to_default=(
        w > 0 and h > 0), upsample_size=args.resize_out_ratio)


# 座標を取得

# p = 関節の番号
    """
    tmp_nose = findPoint(humans, 0)
    nose = tmp_nose

    tmp_heart = findPoint(humans, 1)
    heart = tmp_heart

    right_shoulder = findPoint(humans, 2)
    left_shoulder = findPoint(humans, 5)
    """
    static_image = common.read_imgfile(image, None, None)
    nose_x, nose_y, heart_x, heart_y = TfPoseEstimator.estimate_heart_nose(
        static_image, humans, imgcopy=False)

    print("nose={},{}".format(nose_x, nose_y))  # printは毎回できる。
    print("heart={},{}".format(heart_x, heart_y))
    #position_info = ''
    # position_info = position_info + '鼻(' + str(nose[0]) + ',' + str(
    #   nose[1]) + ';心臓(' + str(heart[0]) + ',' + str(heart[1])+';'

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    time.sleep(20)
