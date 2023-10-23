
#include <typeinfo>

#include "lib.hpp"

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = cv::FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;
cv::Scalar BLACK = Scalar(0, 0, 0);
cv::Scalar BLUE = Scalar(255, 178, 50);
cv::Scalar YELLOW = Scalar(0, 255, 255);
cv::Scalar RED = Scalar(0, 0, 255);  // NOLINT

void write_boxes(const std::string &filename, const std::vector<Rect> &boxes) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cout << "file open failed" << std::endl;
    return;
  }
  for (auto box : boxes) {
    // Do you feel uncomfortable? Do you want to use for_each?
    file << box.x << " " << box.y << " " << box.width << " " << box.height
         << endl;
  }
  file.close();
}

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

vector<cv::Mat> pre_process(const cv::Mat &input_image, Net &net) {  // NOLINT
  // Convert to blob.
  Mat blob;
  vector<int> size = {640, 640};
  Mat blobb;
  cv::resize(input_image, blobb, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0);

  // blobb = input_image.reshape(3, size);
  // std::cout << typeid(blobb).name() << endl;
  // N = blobb.size[0], C = blobb.size[1], H = blobb.size[2], W = blobb.size[3];
  // std::cout << N << "x" << C << "x" << H << "x" << W << endl;
  // cv::imshow("Pre-processed IMage", blobb);
  // cv::waitKey(0);

  cv::dnn::blobFromImage(blobb, blob, 1. / 255., Size(), Scalar(), true, false);
  // int rows = blob.rows;
  // int cols = blob.cols;
  // std::cout << blob.at<float>(0) << endl;
  int N = blob.size[0], C = blob.size[1], H = blob.size[2], W = blob.size[3];
  std::cout << N << "x" << C << "x" << H << "x" << W << endl;
  // std::cout << typeid(blob).name()<< endl;
  net.setInput(blob);

  // Forward propagate.
  vector<Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
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
    if (confidence > CONFIDENCE_THRESHOLD) {  // >=
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
  write_boxes("boxess.txt", boxes);
  // Perform Non-Maximum Suppression and draw predictions.
  vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD,
                    indices);
  // nmsbox(boxes, confidences, NMS_THRESHOLD, indices);
  indices = nmsbox(boxes, confidences, NMS_THRESHOLD);
  cout << "max bounding boxes: " << indices.size() << endl;
  for (auto &&i : indices) {
    // cout <<
    // "===============================bbb================================"
    //         "=="
    //      << endl;
    // cout << "kept boxes:" << boxes[i].x << " " << boxes[i].y << " "
    //      << boxes[i].width << " " << boxes[i].height << endl;
    // cout <<
    // "----------------------------------------------------------------"
    //      << endl;
    // Draw bounding box.
    cv::rectangle(input_image, boxes[i], BLUE, 3 * THICKNESS);
    // Get the label for the class name and its confidence.
    string label = cv::format("%.2f", confidences[i]);
    label = class_name[class_ids[i]] + ":" + label;
    // Draw class labels.
    draw_label(input_image, label, boxes[i].x, boxes[i].y);
  }
  return input_image;
}
