#include "lib.hpp"

vector<int> mask(const vector<float> &vec, const float &threshold) {
  vector<int> resultVec;
  auto len = vec.size();
  // generating indexes where iou is greator than threshold and that are to
  //  be deleted later
  for (int idx = 0; idx < len; ++idx)
    if (vec[idx] > threshold) resultVec.push_back(idx);
  std::cout << "here in mask  input vector len: " << len << std::endl;
  return resultVec;
}

vector<float> overlap(int current_index, int size, const vector<int> &scores,
                      const vector<cv::Rect> &boxes) {
  vector<float> ious;
  for (int i = 1; i < size; i++) {
    cv::Rect intersection_ret = boxes[scores[i]] & boxes[current_index];
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
  int current_index;

  while (sorted_scores.size() > 1) {
    vector<float> ious;
    current_index = sorted_scores[0];
    indicies.push_back(current_index);
    for (auto &&i : sorted_scores) {
      cv::Rect intersection_ret =
          boxes[sorted_scores[i]] & boxes[current_index];
      float area_inter = intersection_ret.area();
      float area_union =
          boxes[i].area() + boxes[current_index].area() - area_inter;
      float iou = area_inter / area_union;
      ious.push_back(iou);
      // ++i;
    }
    std::cout << "ious size" << ious.size() << std::endl;
    auto suppressed = mask(ious, NMS_THRESHOLD);

    // deleteIdxs.push_back(last);
    // suppressed.push_back(current_index);
    std::cout << "in nms before remove by index call: suppressed size "
              << suppressed.size() << std::endl;
    // sorted_scores = RemoveByIndexes(sorted_scores, suppressed);
    std::cout << " in nms before removing sorted_score size: "
              << sorted_scores.size() << std::endl;
    auto offset = 0;
    auto temp = sorted_scores;
    sorted_scores.push_back(current_index);
    for (const auto &idx : suppressed) {
      sorted_scores.erase(next(sorted_scores.begin(), idx + offset));
      // sorted_scores.erase(sorted_scores.begin() + idx + offset);
      offset--;
    }
    // sorted_scores.erase(sorted_scores.begin() + suppressed.size(),
    //                     sorted_scores.end());
    std::cout << " in nms after removing sorted_score size: "
              << sorted_scores.size() << std::endl;
    if (suppressed.size() == 0) break;
  }
}
