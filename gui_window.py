import cv2
import numpy as np
import PySimpleGUI as sg

import argparse

import os
import time

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


parser = argparse.ArgumentParser(description="姿勢推定する")
parser.add_argument("--camera", type=str, default=0)
parser.add_argument("--resize", type=str, default="432x368")
parser.add_argument("--resize-out-ratio", type=float, default=4.0)
parser.add_argument("--model", type=str, default="mobilenet_thin")
args = parser.parse_args()
w, h = model_wh(args.resize)


def findPoint(humans, p):
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


frame_id = 0
first_distance = 0
hunch_sequence = 0

os.makedirs("data/temp", exist_ok=True)
base_path = os.path.join("data/temp", "camera_capture")


sg.theme("LightBlue2")

layout = [
    [
        sg.Text(
            "Realtime movie",
            size=(40, 1),
            justification="center",
            font="Helvetica 20",
            key="-status-",
        )
    ],
    [
        sg.Output(size=(100, 8)),
        sg.Button("Start", size=(10, 1), font="Helvetica 14", key="-start-"),
        sg.Button("Stop", size=(10, 1), font="Helvetica 14", key="-stop-"),
    ],
    # [
    #    sg.Text("Camera number: ", size=(8, 1)),
    #   sg.InputText(default_text="0", size=(4, 1), key="-camera_num-"),
    # ],
    [sg.Image(filename="", key="image", size=(100, 8))],
    # [
    ##   sg.Button("Stop", size=(10, 1), font="Helvetica 14", key="-stop-"),
    # ]
    # sg.Button("Exit", size=(10, 1), font="Helvetica 14", key="-exit-"),
    # ],
]


window = sg.Window("Realtime movie", layout, location=(10, 10), size=(1000, 800))


recording = False

while True:
    event, values = window.read(timeout=20)
    # if event in (None, "-exit-"):
    # break

    if event == "-start-":
        window["-status-"].update("Live")
        # camera_number = int(values["-camera_num-"])
        # cap = cv2.VideoCapture(camera_number, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(0)
        recording = True

    elif event == "-stop-":
        window["-status-"].update("Stop")
        recording = False
        # 幅、高さ　戻り値Float
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(H,W)
        img = np.full((H, W), 0)
        # ndarry to bytes
        imgbytes = cv2.imencode(".png", img)[1].tobytes()
        window["image"].update(data=imgbytes)
        cap.release()
        cv2.destroyAllWindows()

    if recording:
        ret, frame = cap.read()
        if ret is True:
            # imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            # window["image"].update(data=imgbytes)
            # height, width = frame.shape[0], frame.shape[1]

            if h > 0 and w > 0:
                e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
            else:
                e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

            humans = e.inference(
                frame,
                resize_to_default=(w > 0 and h > 0),
                upsample_size=args.resize_out_ratio,
            )

            tmp_nose = findPoint(humans, 0)
            nose = tmp_nose

            tmp_heart = findPoint(humans, 1)
            heart = tmp_heart

            if (nose is None) or (heart is None):
                print("cannot find your body")
            else:
                print("nose={},{}".format(nose[0], nose[1]))  # printは毎回できる。
                print("heart={},{}".format(heart[0], heart[1]))
                distance = (
                    (nose[0] - heart[0]) ** 2 + (nose[1] - heart[1]) ** 2
                ) ** 0.5

                frame_id_str = str(frame_id)

                if frame_id < 3:  # 基準を作る,3回の中で最も大きいやつを基準にする
                    if first_distance < distance:
                        first_distance = distance
                    print("first distance = {}".format(first_distance))
                    cv2.imwrite(
                        "{}_{}.{}".format(base_path, frame_id_str, "jpg"), frame
                    )

                else:
                    if distance < first_distance * 0.9:
                        print("Your back is hancing!")
                        hunch_sequence += 1
                        cv2.imwrite(
                            "{}_{}_{}.{}".format(
                                base_path, frame_id_str, "hunching", "jpg"
                            ),
                            frame,
                        )

                        # if hunch_sequence > 9 and hunch_sequence %  10 == 0 #本当はこっち使いたい
                        if hunch_sequence > 2:
                            cv2.putText(
                                frame,
                                text="Caution!!",
                                org=(250, 300),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=4.0,
                                color=(255, 255, 0),
                                thickness=12,
                                lineType=cv2.LINE_4,
                            )
                        else:
                            cv2.putText(
                                frame,
                                text="hunch back!!",
                                org=(350, 300),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2.0,
                                color=(0, 0, 255),
                                thickness=4,
                                lineType=cv2.LINE_4,
                            )
                    else:
                        hunch_sequence = 0

                print("distance = {},frame_id = {}".format(distance, frame_id))
                print("frame_id = {}".format(frame_id))
                frame_id += 1

            imgbytes = cv2.imencode(".png", frame)[1].tobytes()
            window["image"].update(data=imgbytes)


window.close()
