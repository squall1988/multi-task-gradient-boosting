//
// Created by zebang.zhzb on 2018/6/22.
//

#include <iostream>
#include "MultiTaskUpdater.h"
#include "error.h"

vector<float> MultiTaskUpdater::get_scores(const vector<float> &feature,
                                           const Matrix &gradients,
                                           const vector<int> &task,
                                           const int &task_num,
                                           const vector<int> &sample_index,
                                           float cut_point, float lambda) {
//    if (feature.empty() || sample_index.size() != feature.size()) {
//        return NULL;
//    }
  // sum_g and sum_h for each task,task index from 1 to T
  vector<float> left_sum_g(task_num + 1, 0.0f);
  vector<float> right_sum_g(task_num + 1, 0.0f);
  vector<float> left_sum_h(task_num + 1, 0.0f);
  vector<float> right_sum_h(task_num + 1, 0.0f);
  vector<float> scores(task_num + 1, 0.0f);
  for (int i = 0; i < feature.size(); i++) {
    // 设定成大于cut_point的分到右边，小于cut_point的去左边。
    if (feature[i] >= cut_point) {
      right_sum_g[task[i]] += gradients[i][0];
      right_sum_h[task[i]] += gradients[i][1];
    } else {
      left_sum_g[task[i]] += gradients[i][0];
      left_sum_h[task[i]] += gradients[i][1];
    }
  }
  for (int i = 1; i <= task_num; ++i) {
    float left_score = (left_sum_g[i] * left_sum_g[i]) / (left_sum_h[i] + lambda);
    float right_score = (right_sum_g[i] * right_sum_g[i]) / (right_sum_h[i] + lambda);
    scores[i] = left_score + right_score;
  }
  return scores;

}

float MultiTaskUpdater::get_score(const vector<float> &feature,
                                  const Matrix &gradients,
                                  const vector<int> &sample_index,
                                  float cut_point, float lambda) {
  if (feature.empty() || sample_index.size() != feature.size()) {
    return NODE_SPLIT_ERROR;
  }

  float left_sum_g = 0.0f;
  float right_sum_g = 0.0f;
  float left_sum_h = 0.0f;
  float right_sum_h = 0.0f;

  for (int i = 0; i < feature.size(); i++) {
    // 设定成大于cut_point的分到右边，小于cut_point的去左边。
    if (feature[i] >= cut_point) {
      right_sum_g += gradients[i][0];
      right_sum_h += gradients[i][1];
    } else {
      left_sum_g += gradients[i][0];
      left_sum_h += gradients[i][1];
    }
  }
  float left_score = (left_sum_g * left_sum_g) / (left_sum_h + lambda);
  float right_score = (right_sum_g * right_sum_g) / (right_sum_h + lambda);
  float score = left_score + right_score;
  return score;

}
