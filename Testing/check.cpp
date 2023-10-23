#include <algorithm>
#include <iostream>
#include <vector>

// Define your NMS_THRESHOLD here


int main() {
const float NMS_THRESHOLD = 0.5;
  std::vector<float> scores = {0.9, 0.8, 0.6, 0.3, 0.7};
  std::vector<float> ious = {0.6, 0.4, 0.2, 0.7, 0.5};

  // Check for the size of scores and ious to avoid out-of-bounds access
  if (scores.size() > 1 && ious.size() == scores.size() - 1) {
    scores.erase(scores.begin() + 1,
                 scores.end());  // Equivalent to Python: scores = scores[1:]
    ious.erase(std::remove_if(
                   ious.begin(), ious.end(),
                   [NMS_THRESHOLD](float iou) { return iou < NMS_THRESHOLD; }),
               ious.end());
  }

  // Print the updated scores and ious
  std::cout << "Scores: ";
  for (float score : scores) {
    std::cout << score << " ";
  }
  std::cout << std::endl;

  std::cout << "IoUs: ";
  for (float iou : ious) {
    std::cout << iou << " ";
  }
  std::cout << std::endl;

  return 0;
}
