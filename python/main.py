from model import model
from img_processor import ipa
import cv2


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)


def main():
    process = ipa(
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
    )

    frame = cv2.imread("bus.jpg")

    pre_frame = process.pre_process(frame)
    yolo = model(pre_frame, "./model", "/YOLOv5s.onnx")
    class_ids, predictions = yolo.predict()
    post_processed = process.post_process(class_ids, predictions, frame)
    print(process.__str__())
    print(str(yolo))
    cv2.imshow("Output", post_processed)
    cv2.waitKey(0)
    pass


if __name__ == "__main__":
    main()
