
#include "lib.hpp"  //NOLINT
using cv::Point;
using cv::Rect;
using std::vector;

vector<int> nmsbox(const vector<Rect>& boxes, const vector<float>& scores,
                   const float& threshold) {
  if (boxes.empty()) return vector<int>();

  // auto x1 = GetPointFromRect(boxes, XMIN);
  // auto y1 = GetPointFromRect(boxes, YMIN);
  // auto x2 = GetPointFromRect(boxes, XMAX);
  // auto y2 = GetPointFromRect(boxes, YMAX);
  // compute the area of the bounding boxes and sort the bounding
  // boxes by the bottom-right y-coordinate of the bounding box
  auto area = ComputeArea(boxes);
  auto idxs = argsort(scores);

  int last;
  int i;
  vector<int> pick;
  bool keep = true;
  // keep looping while some indexes still remain in the indexes list
  while (idxs.size() > 0) {  //  !idxs.empty()
    // grab the last index in the indexes list and add the
    // index value to the list of picked indexes
    last = idxs.size() - 1;
    i = idxs[last];
    // cout << i << "th: saved index" << endl;
    // if (keep)
    pick.push_back(i);
    vector<float> overlap;
    // find the largest (x, y) coordinates for the start of
    // the bounding box and the smallest (x, y) coordinates
    // for the end of the bounding box
    auto idxsWoLast = RemoveLast(idxs);
    // iou calculation -> not sure: error here in calculating intersection area.
    for (auto box_index = 0; box_index < idxsWoLast.size(); ++box_index) {
      auto inter_rect = boxes[i] & boxes[box_index];
      float area_inter = inter_rect.area();
      auto area_union =
          boxes[i].area() + boxes[box_index].area() - inter_rect.area();
      float iou = area_inter / area_union;
      // keep = iou <= threshold-.2;
      // cout << iou << " iou compared with box at index " << i << endl;
      overlap.push_back(iou);
    }
    // auto xx1 = Maximum(x1[i], CopyByIndexes(x1, idxsWoLast));
    // auto yy1 = Maximum(y1[i], CopyByIndexes(y1, idxsWoLast));
    // auto xx2 = Minimum(x2[i], CopyByIndexes(x2, idxsWoLast));
    // auto yy2 = Minimum(y2[i], CopyByIndexes(y2, idxsWoLast));

    // // compute the width and height of the bounding box
    // auto w = Maximum(0, Subtract(xx2, xx1));
    // auto h = Maximum(0, Subtract(yy2, yy1));

    // // compute the ratio of overlap
    // auto overlap = Divide(Multiply(w, h), CopyByIndexes(area, idxsWoLast));

    // delete all indexes from the index list that have
    auto deleteIdxs = WhereLarger(overlap, threshold);
    deleteIdxs.push_back(last);
    auto idxss = RemoveByIndexes(idxs, deleteIdxs);
    // auto idxss = remove_if(idxs.begin(), idxs.end(), WhereLarger);
    // idxs.clear();
    idxs = idxss;
    cout << idxs.size() << endl;
    for (auto&& j : deleteIdxs) {
      cout << "removed boxes:" << boxes[j].x << " " << boxes[j].y << " "
           << boxes[j].width << " " << boxes[j].height << endl;
      cout << "----------------------------------------------------------------"
           << endl;
    }
    for (auto&& jk : idxs) {
      cout << "----------------------------------------------------------------"
           << endl;
      cout << "----------------------------------------------------------------"
           << endl;
      cout << "remaining boxes:" << boxes[jk].x << " " << boxes[jk].y << " "
           << boxes[jk].width << " " << boxes[jk].height << endl;
      cout << "----------------------------------------------------------------"
           << endl;
    }

    cout << "================================================================="
         << endl;
    cout << "kept boxes:" << boxes[i].x << " " << boxes[i].y << " "
         << boxes[i].width << " " << boxes[i].height << endl;
    cout << "----------------------------------------------------------------"
         << endl;
  }

  return pick;
}

vector<float> GetPointFromRect(const vector<Rect>& rect,
                               const PointInRectangle& pos) {
  vector<float> points;

  for (const auto& p : rect) {
    if (pos == 0) {
      points.push_back(p.x);
    }

    if (pos == 1) {
      points.push_back(p.y);
    }

    if (pos == 2) {
      points.push_back(p.x + (p.width / 2));
    }

    if (pos == 3) {
      points.push_back(p.y + (p.height / 2));
    }

    return points;
  }
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
  sort(idx.begin(), idx.end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });

  return idx;
}

vector<float> Maximum(const float& num, const vector<float>& vec) {
  auto maxVec = vec;
  auto len = vec.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] < num) maxVec[idx] = num;

  return maxVec;
}

vector<float> Minimum(const float& num, const vector<float>& vec) {
  auto minVec = vec;
  auto len = vec.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    if (vec[idx] > num) minVec[idx] = num;

  return minVec;
}

vector<float> CopyByIndexes(const vector<float>& vec, const vector<int>& idxs) {
  vector<float> resultVec;

  for (const auto& idx : idxs) resultVec.push_back(vec[idx]);

  return resultVec;
}

vector<int> RemoveLast(const vector<int>& vec) {
  auto resultVec = vec;
  resultVec.erase(resultVec.end() - 1);
  return resultVec;
}

vector<float> Subtract(const vector<float>& vec1, const vector<float>& vec2) {
  vector<float> result;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    result.push_back(vec1[idx] - vec2[idx]);

  return result;
}

vector<float> Multiply(const vector<float>& vec1, const vector<float>& vec2) {
  vector<float> resultVec;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] * vec2[idx]);

  return resultVec;
}

vector<float> Divide(const vector<float>& vec1, const vector<float>& vec2) {
  vector<float> resultVec;
  auto len = vec1.size();

  for (decltype(len) idx = 0; idx < len; ++idx)
    resultVec.push_back(vec1[idx] / vec2[idx]);

  return resultVec;
}

vector<int> WhereLarger(const vector<float>& vec, const float& threshold) {
  vector<int> resultVec;
  auto len = vec.size();
  // decltype(len)
  for (int idx = 0; idx < len; idx++)
    if (vec[idx] >= threshold) resultVec.push_back(idx);

  return resultVec;
}

vector<int> RemoveByIndexes(const vector<int>& vec, const vector<int>& idxs) {
  auto resultVec = vec;
  auto offset = 0;

  for (const auto& idx : idxs) {
    resultVec.erase(resultVec.begin() + idx + offset);
    cout << "Removed index" << idx << endl;
    offset -= 1;
  }
  cout << "----------------------------------------------------------------"
       << endl;

  return resultVec;
}
