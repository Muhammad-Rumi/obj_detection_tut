import cv2
import numpy as np


class features:
    class_ids = []
    confidences = []
    boxes = []


class ipa:
    # Constants.

    def __init__(
        self,
        INPUT_WIDTH,
        INPUT_HEIGHT,
        SCORE_THRESHOLD,
        NMS_THRESHOLD,
        CONFIDENCE_THRESHOLD,
        FONT_FACE,
        FONT_SCALE,
        THICKNESS,
        BLACK,
        BLUE,
        YELLOW,
    ) -> None:
        self.INPUT_WIDTH = INPUT_WIDTH
        self.INPUT_HEIGHT = INPUT_HEIGHT
        self.SCORE_THRESHOLD = SCORE_THRESHOLD
        self.NMS_THRESHOLD = NMS_THRESHOLD
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.FONT_FACE = FONT_FACE
        self.FONT_SCALE = FONT_SCALE
        self.THICKNESS = THICKNESS
        self.BLACK = BLACK
        self.BLUE = BLUE
        self.YELLOW = YELLOW

    def draw_label(self, im, label, x, y):
        text_size = cv2.getTextSize(
            label, self.FONT_FACE, self.FONT_SCALE, self.THICKNESS
        )
        dim, baseline = text_size[0], text_size[1]
        # Use text size to create a BLACK rectangle.
        cv2.rectangle(
            im,
            (x, y),
            (x + dim[0], y + dim[1] + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        # Display text inside the rectangle.
        cv2.putText(
            im,
            label,
            (x, y + dim[1]),
            self.FONT_FACE,
            self.FONT_SCALE,
            self.YELLOW,
            self.THICKNESS,
            cv2.LINE_AA,
        )

    def pre_process(self, frame):
        assert frame is not None, "Image not found"
        image_height, image_width = frame.shape[:2]
        assert frame.shape[:2] == (1080, 810), f"wrong Size {frame.shape[:2]}"
        # Resizing factor.
        self.x_factor = image_width / self.INPUT_WIDTH
        self.y_factor = image_height / self.INPUT_HEIGHT
        # assert 0, f'{self.x_factor}'
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # remove and run the model
        img = cv2.resize(frame, (640, 640)) / 255.0

        img = np.transpose(img, axes=[2, 0, 1])
        cv2.imshow('red color',img[1,:,:])
        cv2.waitKey(0)
        img = np.reshape(img, (1, 3, 640, 640))
        # img = np.reshape(img, (16, 3, 640, 640))  # change most assending
        return img

    def filtering(self, pred):
        if pred[4] > self.CONFIDENCE_THRESHOLD:
            class_id = np.argmax(pred[5:])
            if pred[5 + class_id] > self.SCORE_THRESHOLD:
                return True
            else:
                return False
        return False

    def over_lap(self, boxes_a: np.ndarray, boxes_b: np.ndarray):
        # assert 0 , f'{boxes_b.T[:,:2].shape}'
        box_area = lambda box: box[3] * box[2]
        area_a = boxes_a[3] * boxes_a[2]
        area_b = np.array(
            list(map(box_area, boxes_b.T))
        )  # area of all the boxes other than th curren box_idx

        top_left = np.maximum(
            np.reshape(boxes_a[:2], (2, 1)),
            boxes_b[:2, :],
        )
        bottom_right = np.minimum(
            np.reshape(boxes_a[:2] + (boxes_a[2:]), (2, 1)),
            boxes_b[:2, :] + boxes_b[2:, :],
        )

        area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 0)
        # assert 0, f'{area_b}'
        area_union = area_a + area_b - area_inter
        # assert 0, f'{area_union.shape}'
        return area_inter / area_union

    def nms(self):
        indicies = list()
        self.boxes, self.confidences, self.class_id = self.rescale(
            self.filtered_by_confidence_score
        )
        # assert 0 , f'{self.confidences.shape}'
        # assert self.confidences.shape == (44,) and self.boxes.shape == (
        #     4,
        #     44,
        # ), f"check again rescale function {self.boxes.shape}, {len(self.confidences)}"
        scores = np.argsort(self.confidences)[::-1]
        # assert len(scores) == 44
        # assert 0, f"{boxes[:, scores[0:]].shape}"
        while len(scores) > 0:
            current_box_idx = scores[0]
            indicies.append(current_box_idx)

            curr_box = self.boxes[:, current_box_idx]
            remaining = self.boxes[:, scores[1:]]
            ious = self.over_lap(curr_box, remaining)

            scores = scores[1:][ious < self.NMS_THRESHOLD]
            print(scores.shape)
        print(indicies)
        return indicies

    def rescale(self, pred):
        assert type(pred) == np.ndarray, f"{type(pred)}"

        box_dems = pred[:, :4]
        # assert 0, f'{box_dems}'
        # assert 0, f"{box_dems[:,0].shape}"
        # assert 0, pred[:]
        left = (box_dems[:, 0] - box_dems[:, 2] / 2) * self.x_factor
        top = (box_dems[:, 1] - box_dems[:, 3] / 2) * self.y_factor
        width = box_dems[:, 2] * self.x_factor
        height = box_dems[:, 3] * self.y_factor
        # assert 0, f'{len(pred[:,5:])}'
        return (
            np.array([left, top, width, height]),
            pred[:, 4],
            np.argmax(pred[:, 5:], axis=1),
        )

    def post_process(self, classes, predictions: list, input_image):
        # assert 0 , f'{input_image.shape}'
        # Rows.
        # rows = predictions[0].shape[1]  # incase of yolov5 must me 25200
        # assert rows == 25200, f"{rows}"
        # assert len(predictions[0][0]) == 25200  # yolo specific
        filtered_by_confidence_score = filter(self.filtering, predictions[0][0])
        self.filtered_by_confidence_score = np.array(
            [i for i in filtered_by_confidence_score]
        )
        indicies = self.nms()
        # assert(0), f"{self.boxes[:,0][1]}"
        for i in indicies:
            box = self.boxes[:, i]
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            # assert 0, f'{height}'
            # Draw bounding box.
            cv2.rectangle(
                input_image,
                (left, top),
                (left + width, top + height),
                self.BLUE,
                3 * self.THICKNESS,
            )
            # assert 0, classes[self.class_id[0]]

            # Class label.
            label = "{}:{:.2f}".format(classes[self.class_id[i]], self.confidences[i])
            # Draw label.
            self.draw_label(input_image, label, left, top)
        # print(self.boxes.shape)
        return input_image

    def __str__(self, flag=""):
        separator = (
            "\n==================================================================\n"
        )
        additional_info = (
            f"x_factor: {self.x_factor}<|----|> y_factor: {self.y_factor}\n"
        )
        # class_scores = self.class_scores
        return separator + additional_info + separator
