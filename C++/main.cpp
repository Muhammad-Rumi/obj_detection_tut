#include "lib.hpp"
// cv::Scalar BLACK = Scalar(0, 0, 0);
// cv::Scalar BLUE = Scalar(255, 178, 50);
// cv::Scalar YELLOW = Scalar(0, 255, 255);
// cv::Scalar RED = Scalar(0, 0, 255);

int main() {
  // Load class list.
  std::vector<string> class_list = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush "};

  cv::Mat frame;
  frame = cv::imread("../bus.jpg");
  // Load model.
  cv::dnn::Net net;
  net = cv::dnn::readNet("../model/YOLOv5s.onnx");
  vector<cv::Mat> detections;  // Process the image.
  detections = pre_process(frame, net);
  // std::cout << detections.size() << endl;

  cv::Mat frame_clone = frame.clone();
  cv::Mat img = post_process(frame_clone, detections, class_list);
  // Put efficiency information.
  // The function getPerfProfile returns the overall time for inference(t)
  // and the timings for each of the layers(in layersTimes).
  // vector<double> layersTimes;
  // double freq = cv::getTickFrequency() / 1000;
  // double t = net.getPerfProfile(layersTimes) / freq;
  // string label = cv::format("Inference time : %.2f ms", t);
  // cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
  // cv::dnn::
  // cv::namedWindow("thewindow", cv::WINDOW_NORMAL);
  cv::imshow("thewindow", img);
  cv::waitKey(0);
  return 0;
}
