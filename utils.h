//
// Created by squall on 18-6-11.
//
#pragma once

#ifndef MTREE_UTILS_H
#define MTREE_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <map>
#include "error.h"

#define ERROR_CODE_CHECK(x) \
  if ((x) != 0 ) {\
  cout << "This is error" << endl; \
  return -1; \
}

#define THREAD_NUM 4

namespace common {

using namespace std;
typedef vector<vector<float> > Matrix;

template<typename LossObj>
inline Matrix get_default_gradients(const vector<int> &label, float default_score) {
  /*
   * \belief: before the first round, we need to get the default gradients.
   */
  LossObj loss;
}

inline std::vector<std::string> Split(const std::string &s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

inline float RMSE(const vector<float> &label, const vector<float> &pred) {
  if (label.size() != pred.size() || label.empty() || pred.empty()) {
    return RMSE_ERROR;
  }
  float sum_error = 0.0f;
  for (int i = 0; i < label.size(); ++i) {
    float tmp = (float) label[i] - pred[i];
    sum_error += tmp * tmp;
  }
  return sqrt(sum_error / label.size());
}

inline float nrMSE(const vector<float> &label, const vector<float> &pred) {
  if (label.size() != pred.size() || label.empty() || pred.empty()) {
    return nrMSE_ERROR;
  }
  float sum_error = 0.0f;
  float Max = label[0], Min = label[0];
  for (int i = 0; i < label.size(); ++i) {
    Max = label[i] > Max ? label[i] : Max;
    Min = label[i] < Min ? label[i] : Min;
    float tmp = (float) label[i] - pred[i];
    sum_error += tmp * tmp;
  }
  return sqrt(sum_error / label.size()) / (Max - Min);
}

inline float BinaryLogLoss(const vector<float> &label, const vector<float> &pred) {
  if (label.size() != pred.size() || label.empty() || pred.empty()) {
    return LOG_LOSS_ERROR;
  }
  float sum_error = 0.0f;
  for (int i = 0; i < label.size(); ++i) {
    bool flag = false;
    if (label[i] <= 0) {
      if (1.0f - (1.0f / (1.0f + std::exp(-pred[i]))) > 1e-15f) {
        flag = true;
        sum_error += -std::log(1.0f - (1.0f / (1.0f + std::exp(-pred[i]))));
      }
    } else {
      if ((1.0f / (1.0f + std::exp(-pred[i]))) > 1e-15f) {
        flag = true;
        sum_error += -std::log((1.0f / (1.0f + std::exp(-pred[i]))));
      }
    }
    if (!flag) {
      sum_error += -std::log(1e-15f);
    }
  };
  return sum_error / label.size();
}

inline float ApproximateAUC(const vector<float> &label, const vector<float> &pred) {
  if (label.size() != pred.size() || label.empty() || pred.empty()) {
    return AUC_ERROR;
  }
  int n = label.size();
  int posNum = 0;
  int negNum = 0;
  for (int i = 0; i < n; ++i) {
    if (int(label[i]) == 1) {
      posNum += 1;
    } else {
      negNum += 1;
    }
  }
  int n_bins = 100;
  vector<int> pos_histogram(n_bins, 0);
  vector<int> neg_histogram(n_bins, 0);
  float bin_width = 1.0 / n_bins;
  for (int i = 0; i < n; ++i) {
    int nth_bin = (int) ((1.0f / (1.0f + std::exp(-pred[i]))) / bin_width);
    if (label[i] == 1) {
      pos_histogram[nth_bin] += 1;
    } else {
      neg_histogram[nth_bin] += 1;
    }
  }

  int accumulated_neg = 0;
  int satisfied_pair = 0;
  for (int i = 0; i < n_bins; ++i) {
    satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5);
    accumulated_neg += neg_histogram[i];
  }
  float auc = (float) satisfied_pair / (posNum * negNum);
  return auc;
}

inline Matrix transpose(const Matrix &mat) {
  Matrix mat_tran(mat[0].size(), vector<float>(mat.size(), 0));
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < mat[i].size(); ++j) {
      mat_tran[j][i] = mat[i][j];
    }
  }
  return mat_tran;
}

inline bool cmp(const pair<float, float> &a, const pair<float, float> &b) {
  if (a.first == b.first) return a.second < b.second;
  else return a.first < b.first;
}



inline bool auc_cmp(const pair<float, float> &a, const pair<float, float> &b){
  return a.second < b.second;
}

inline double AUC(const vector<float> &label, const vector<float> &pred) {
  if (label.size() != pred.size() || label.empty() || pred.empty()) {
    return AUC_ERROR;
  }
  int n = label.size();
  int64_t posNum = 0;
  int64_t negNum = 0;
  for (int i = 0; i < n; ++i) {
    if (label[i] == 1) {
      posNum += 1;
    } else {
      negNum += 1;
    }
  }
  int64_t total_pair = posNum * negNum;
  vector<pair<float, float>>label_pred;
  for(int i=0;i<n;++i) {
    label_pred.emplace_back(make_pair(label[i],pred[i]));
  }
  sort(label_pred.begin(), label_pred.end(), auc_cmp);
  int64_t accumulated_neg = 0;
  int64_t satisfied_pair = 0;
  for(int i=0;i<n;++i) {
    if(label_pred[i].first == 1) {
      satisfied_pair += accumulated_neg;
    }else{
      accumulated_neg += 1;
    }
  }
  cout << satisfied_pair << ", "<< total_pair << " , " << static_cast<double>(total_pair) << endl;
  return satisfied_pair / static_cast<double>(total_pair);
}

inline void fill_index(vector<int>& src) {
  std::generate(src.begin(), src.end(), [n=0]() mutable {return n++;});
}

} // common



#endif //MTREE_UTILS_H
