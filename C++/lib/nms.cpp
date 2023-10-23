
#include "lib.hpp"  //NOLINT
using cv::Point;
using cv::Rect;
using std::vector;
using namespace std;
vector<int> nmsbox(const vector<Rect>& boxes, const vector<float>& scores,
                   const float& threshold) {
  if (boxes.empty()) return vector<int>();

  auto area = ComputeArea(boxes);
  auto idxs = argsort(scores);

  int last;
  int i;
  vector<int> pick;
  float adaptive_threshold = threshold;
  // keep looping while some indexes still remain in the indexes list
  for (int last = 0; last < idxs.size(); ++last) {
    cout << "outer for loop" << endl;
    i = idxs[last];
    cout << "outer for loop value for i:" << i << endl;
    bool keep = true;
    for (int box_index = 0; (box_index < pick.size()) && keep; ++box_index) {
      cout << "inner for loop" << endl;
      const int kept_index = pick[box_index];
      auto inter_rect = boxes[i] & boxes[kept_index];
      double area_inter = inter_rect.area();
      auto area_union =
          boxes[i].area() + boxes[kept_index].area() - inter_rect.area();
      double iou = area_inter / area_union;
      keep = iou <= adaptive_threshold;
    }

    if (keep) {
      std::cout << i << " pick box & size: " << pick.size() << endl;
      pick.push_back(i);
    }
  }

  return pick;
}



vector<float> ComputeArea(const vector<Rect>& x1) {
  vector<float> area;
  auto len = x1.size();

  for (decltype(len) idx = 0; idx < len; ++idx) {
    auto tmpArea = x1[idx].area();
    area.push_back(tmpArea);
  }

  return area;
}

template <typename T>
vector<int> argsort(const vector<T>& v) {
  // initialize original index locations
  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] > v[i2]; });

  return idx;
}


