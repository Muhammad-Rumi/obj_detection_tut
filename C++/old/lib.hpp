// "Copywite [2023] <Muhammad Rumi>"  // NOLINT
#pragma once
#include <algorithm>  //NOLINT
#include <fstream>    // NOLINT
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Namespaces.
using namespace cv;       // NOLINT
using namespace std;      // NOLINT
using namespace cv::dnn;  // NOLINT

void draw_label(const cv::Mat &input_image, string label, int left, int top);
vector<cv::Mat> pre_process(const cv::Mat &input_image, Net &net);

cv::Mat post_process(Mat &input_image, const vector<Mat> &outputs,  // NOLINT
                     const vector<string> &class_name);

enum PointInRectangle { XMIN, YMIN, XMAX, YMAX };

std::vector<int> nmsbox(const std::vector<Rect> &, const std::vector<float> &,
                        const float &);

std::vector<float> GetPointFromRect(const std::vector<Rect> &,
                                    const PointInRectangle &);

std::vector<float> ComputeArea(const std::vector<Rect> &x1);

template <typename T>
std::vector<int> argsort(const std::vector<T> &v);

std::vector<float> Maximum(const float &, const std::vector<float> &);

std::vector<float> Minimum(const float &, const std::vector<float> &);

std::vector<float> CopyByIndexes(const std::vector<float> &,
                                 const std::vector<int> &);

std::vector<int> RemoveLast(const std::vector<int> &);

std::vector<float> Subtract(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Multiply(const std::vector<float> &,
                            const std::vector<float> &);

std::vector<float> Divide(const std::vector<float> &,
                          const std::vector<float> &);

std::vector<int> WhereLarger(const std::vector<float> &, const float &);

std::vector<int> RemoveByIndexes(const std::vector<int> &,
                                 const std::vector<int> &);

std::vector<cv::Rect> BoxesToRectangles(
    const std::vector<std::vector<float>> &);

template <typename T>
std::vector<T> FilterVector(const std::vector<T> &, const std::vector<int> &);

extern const float INPUT_WIDTH;
extern const float INPUT_HEIGHT;
extern const float SCORE_THRESHOLD;
extern const float NMS_THRESHOLD;
extern const float CONFIDENCE_THRESHOLD;

// Text parameters.
extern const float FONT_SCALE;
extern const int FONT_FACE;
extern const int THICKNESS;

// Colors.
extern cv::Scalar BLACK;
extern cv::Scalar BLUE;
extern cv::Scalar YELLOW;
extern cv::Scalar RED;
