import os
import shutil
import numpy as np
from six import BytesIO
from PIL import Image

from tensorflow_hub import load as hub_load

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


class ObjectDetector:
    def __init__(self, model_name: str) -> None:

        self.MODELS = {
            "SSD MobileNet V2 FPNLite 320x320": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
            "CenterNet ResNet50 Keypoints 512x512": "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1",
            "CenterNet HourGlass104 Keypoints 512x512": "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1",
        }

        self.PATHS = {
            "PATH_TO_LABELS": "object_detection/data/mscoco_label_map.pbtxt"
        }

        self.COCO17_HUMAN_POSE_KEYPOINTS = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 6),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

        hub_model = self.MODELS[model_name]
        self.model = hub_load(hub_model)
        self.categoryIndex = label_map_util.create_category_index_from_labelmap(
            self.PATHS["PATH_TO_LABELS"], use_display_name=True
        )

    def ImgToNpArray(self, img) -> np.ndarray:
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
        img: the byte file image

        Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
        """
        image = Image.open(BytesIO(img))
        (im_width, im_height) = image.size
        return (
            np.array(image.getdata())
            .reshape((1, im_height, im_width, 3))
            .astype(np.uint8)
        )

    def Inference(self, img_np: np.ndarray) -> dict:
        # running inference
        results = self.model(img_np)

        # different object detection models have additional results
        # all of them are explained in the documentation
        result = {key: value.numpy() for key, value in results.items()}
        return result

    def DetectionImage(self, result: dict, img: np.ndarray):
        label_id_offset = 0
        image_np_with_detections = img

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if "detection_keypoints" in result:
            keypoints = result["detection_keypoints"][0]
            keypoint_scores = result["detection_keypoint_scores"][0]

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections[0],
            result["detection_boxes"][0],
            (result["detection_classes"][0] + label_id_offset).astype(int),
            result["detection_scores"][0],
            self.categoryIndex,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.30,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=self.COCO17_HUMAN_POSE_KEYPOINTS,
        )

        temp_img = Image.fromarray(image_np_with_detections[0])
        temp_img.save("test_img.jpeg")
        if not os.path.exists("../temp/images"):
            os.makedirs("../temp/images")
        elif os.path.exists("../temp/images/test_img.jpeg"):
            os.remove("../temp/images/test_img.jpeg")
        shutil.move("test_img.jpeg", "../temp/images")

        return "../temp/images/test_img.jpeg"
