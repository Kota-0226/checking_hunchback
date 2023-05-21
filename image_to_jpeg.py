import cv2
import numpy as np
import base64

def to_html_img_tag(img: np.ndarray, attributes: dict = {}) -> str:
    """
    Open CV image to HTML image tag

    Parameters
    ----------
    img : cv2.Mat
        Open CV image
    attributes: dict
        Attributes

    Returns
    -------
    """
    attributes_str = ""
    for key, val in attributes.items():
        if type(val) is str:
            attributes_str += f' {key}="{val}"'
        elif type(val) is float or type(val) is int:
            attributes_str += f" {key}={val}"
        else:
            raise Exception("Incorrect type attributes")

    cnt = cv2.imencode(".png", img)[1]
    dat: str = base64.encodebytes(cnt).decode("utf-8")
    return f'<img src="data:image/png;base64,{dat}"{attributes_str}>'

if __name__ == '__main__':
    pass
