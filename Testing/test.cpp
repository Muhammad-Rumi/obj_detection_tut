#include "lib.hpp"

void draw_label(const cv::Mat &input_image, string label, int left, int top) {
  // Display the label at the top of the bounding box.
  int baseLine;
  Size label_size =
      cv::getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
  top = max(top, label_size.height);
  // Top left corner.
  Point tlc = cv::Point(left, top);
  // Bottom right corner.
  Point brc =
      cv::Point(left + label_size.width, top + label_size.height + baseLine);
  // Draw white rectangle.
  cv::rectangle(input_image, tlc, brc, BLACK, FILLED);
  // Put the label on the black rectangle.
  cv::putText(input_image, label, Point(left, top + label_size.height),
              FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

// Now, elementsWithIndices contains the el
vector<cv::Mat> pre_process(const cv::Mat &input_image, Net &net) {  // NOLINT
  // Convert to blob.
  Mat blob;
  cv::dnn::blobFromImage(input_image, blob, 1. / 255.,
                         Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true,
                         false);

  net.setInput(blob);

  // Forward propagate.
  vector<Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
}

template <typename T>
vector<int> argsort(const vector<T> &v) {
  // initialize original index locations
  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}

vector<int> mask(const vector<float> &vec, const float &threshold) {
  vector<int> resultVec;
  auto len = vec.size();
  // generating indexes where iou is greator than threshold and that are to
  //  be deleted later
  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > threshold) resultVec.push_back(idx);
  std::cout << "here in mask  input vector len: " << len << std::endl;
  return resultVec;
}

vector<int> RemoveByIndexes(const vector<int> &vec, const vector<int> &idxs) {
  auto resultVec = vec;
  auto offset = 0;
  auto index = idxs;
  sort(index.rbegin(), index.rend());
  for (const auto &idx : idxs) {
    resultVec.erase(resultVec.begin() + idx);
    // offset -= 1;
  }
  std::cout << "here in remove indexes after removal len: " << resultVec.size()
            << std::endl;
  return resultVec;
}
vector<float> overlap(int current_index, int size,
                      const vector<cv::Rect> &boxes) {
  vector<float> ious;
  for (int i = current_index + 1; i < size; i++) {
    cv::Rect intersection_ret = boxes[i] & boxes[current_index];
    float area_inter = intersection_ret.area();
    float area_union =
        boxes[i].area() + boxes[current_index].area() - area_inter;
    float iou = area_inter / area_union;
    ious.push_back(iou);
    // ++i;
  }
  return ious;
}

void nmsbox(const std::vector<cv::Rect> &boxes,
            const std::vector<float> &scores, const float nms_threshold,
            std::vector<int> &indicies) {  // NOLINT
  std::vector<int> sorted_scores = argsort(scores);
  vector<int> pick;
  vector<float> ious;
  int current_index;

  while (sorted_scores.size() > 0) {
    current_index = sorted_scores[0];
    indicies.push_back(current_index);
    ious = overlap(current_index, sorted_scores.size(), boxes);
    std::cout << "ious size" << ious.size() << std::endl;
    auto suppressed = mask(ious, NMS_THRESHOLD);

    // deleteIdxs.push_back(last);
    suppressed.push_back(current_index);
    std::cout << "in nms before remove by index call: suppressed size "
              << suppressed.size() << std::endl;
    sorted_scores = RemoveByIndexes(sorted_scores, suppressed);
    std::cout << " in nms after removing sorted_score size: "
              << sorted_scores.size() << std::endl;
  }
}

cv::Mat post_process(Mat &input_image, const vector<Mat> &outputs,  // NOLINT
                     const vector<string> &class_name) {
  // Initialize vectors to hold respective outputs while unwrapping detections.
  vector<int> class_ids;
  vector<float> confidences;
  vector<Rect> boxes;
  // Resizing factor.
  float x_factor = input_image.cols / INPUT_WIDTH;
  float y_factor = input_image.rows / INPUT_HEIGHT;
  float *data = reinterpret_cast<float *>(outputs[0].data);
  const int dimensions = 85;
  // 25200 for default size 640.
  const int rows = 25200;
  // Iterate through 25200 detections.
  for (int i = 0; i < rows; ++i) {
    float confidence = data[4];
    // Discard bad detections and continue.
    if (confidence >= CONFIDENCE_THRESHOLD) {
      float *classes_scores = data + 5;
      // Create a 1x85 Mat and store class scores of 80 classes.
      cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
      // Perform minMaxLoc and acquire the index of best class  score.
      cv::Point class_id;
      double max_class_score;
      cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      // Continue if the class score is above the threshold.
      if (max_class_score > SCORE_THRESHOLD) {
        // Store class ID and confidence in the pre-defined respective vectors.
        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        // Center.
        float cx = data[0];
        float cy = data[1];
        // Box dimension.
        float w = data[2];
        float h = data[3];
        // Bounding box coordinates.
        int left = int((cx - 0.5 * w) * x_factor);
        int top = int((cy - 0.5 * h) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);
        // Store good detections in the boxes vector.
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    // Jump to the next row.
    data += 85;
  }
  // Perform Non-Maximum Suppression and draw predictions.
  vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
  indices);
  cout << "max bounding boxes: " << indices.size() << endl;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    Rect box = boxes[idx];
    int left = box.x;
    int top = box.y;
    int width = box.width;
    int height = box.height;
    std::cout << box << endl;
    // Draw bounding box.
    cv::rectangle(input_image, cv::Point(left, top),
                  cv::Point(left + width, top + height), BLUE, 3 * THICKNESS);
    // Get the label for the class name and its confidence.
    string label = cv::format("%.2f", confidences[idx]);
    label = class_name[class_ids[idx]] + ":" + label;
    // Draw class labels.
    draw_label(input_image, label, left, top);
  }
  return input_image;
}

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
  cv::Mat frame_clone = frame.clone();
  cv::Mat img = post_process(frame_clone, detections, class_list);
  // Put efficiency information.
  // The function getPerfProfile returns the overall time for     inference(t)
  // and the timings for each of the layers(in layersTimes).
  vector<double> layersTimes;
  double freq = cv::getTickFrequency() / 1000;
  double t = net.getPerfProfile(layersTimes) / freq;
  string label = cv::format("Inference time : %.2f ms", t);
  cv::putText(img, label, cv::Point(20, 40), FONT_FACE, FONT_SCALE, RED);
  cv::imshow("Output", img);
  cv::waitKey(0);
  return 0;
}
