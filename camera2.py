import cv2
import numpy as np

import argparse
import os
import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import slackweb

# from gui_window import findPoint

parser = argparse.ArgumentParser(description="姿勢推定する")
parser.add_argument("--camera", type=str, default=0)
parser.add_argument("--resize", type=str, default="432x368")
parser.add_argument("--resize-out-ratio", type=float, default=4.0)
parser.add_argument("--model", type=str, default="mobilenet_thin")
args = parser.parse_args()
w, h = model_wh(args.resize)


slack = slackweb.Slack(
    url="https://hooks.slack.com/services/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"
)
slack.notify(text="背中曲がってるよー！")


class VideoCamera(object):
    frame_id = 0
    first_distance = 0
    hunch_sequence = 0

    os.makedirs("data/temp", exist_ok=True)
    base_path = os.path.join("data/temp", "camera_capture")

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FPS, 1)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        # self.quality = 25
        # self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

    def __del__(self):
        self.video.release()

    def findPoint(self, humans, p):
        for human in humans:
            try:
                body_part = human.body_parts[p]
                parts = [0, 0]

                # 座標を整数に切り上げで置換
                parts[0] = int(body_part.x * w + 0.5)
                parts[1] = int(body_part.y * h + 0.5)
                # parts = [x座標, y座標]
                return parts
            except:
                print("Not found")

    def get_frame(self):
        success, image = self.video.read()

        frame = None  # frame変数を初期化

        if h > 0 and w > 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

        humans = e.inference(
            image,
            resize_to_default=(w > 0 and h > 0),
            upsample_size=args.resize_out_ratio,
        )

        nose = self.findPoint(humans, 0)

        heart = self.findPoint(humans, 1)

        if (nose is None) or (heart is None):
            print("cannot find your body")
        else:
            print("nose={},{}".format(nose[0], nose[1]))  # printは毎回できる。
            print("heart={},{}".format(heart[0], heart[1]))
            distance = ((nose[0] - heart[0]) ** 2 + (nose[1] - heart[1]) ** 2) ** 0.5

            frame_id_str = str(self.frame_id)

            if self.frame_id < 3:  # 基準を作る,3回の中で最も大きいやつを基準にする
                if self.first_distance < distance:
                    self.first_distance = distance
                print("first distance = {}".format(self.first_distance))
                cv2.imwrite(
                    "{}_{}.{}".format(self.base_path, frame_id_str, "jpg"), frame
                )

            else:
                if distance < self.first_distance * 0.92:
                    print("Your back is hancing!")
                    self.hunch_sequence += 1
                    cv2.imwrite(
                        "{}_{}_{}.{}".format(
                            self.base_path, frame_id_str, "hunching", "jpg"
                        ),
                        image,
                        # frame,
                    )

                    # if hunch_sequence > 9 and hunch_sequence %  10 == 0 #本当はこっち使いたい
                    if self.hunch_sequence > 2:
                        cv2.putText(
                            image,
                            text="Caution!!",
                            org=(120, 300),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.0,
                            color=(0, 255, 255),
                            thickness=12,
                            lineType=cv2.LINE_4,
                        )
                        slack.notify(text=notify.txt)
                    else:
                        cv2.putText(
                            image,
                            text="hunch back!!",
                            org=(200, 300),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2.0,
                            color=(0, 0, 255),
                            thickness=4,
                            lineType=cv2.LINE_4,
                        )
                else:
                    self.hunch_sequence = 0

            print("distance = {},frame_id = {}".format(distance, VideoCamera.frame_id))
            print("frame_id = {}".format(VideoCamera.frame_id))
            VideoCamera.frame_id += 1
            # self.frame_id += 1

        # imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        # window["image"].update(data=imgbytes)

        # return jpeg.tobytes()
        # return frame.tobytes()

        ret, frame = cv2.imencode(".jpg", image)
        return frame.tobytes()
