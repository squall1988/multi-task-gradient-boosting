//
// Created by zebang.zhzb on 2018/6/22.
//

#ifndef MTREE_MULTITASKUPDATER_H
#define MTREE_MULTITASKUPDATER_H

#include "Updater.h"

class MultiTaskUpdater : public Updater {
 public:
  MultiTaskUpdater() {}
  ~MultiTaskUpdater() {}
  float get_score(const vector<float> &feature,
                  const Matrix &gradients,
                  const vector<int> &sample_index,
                  float cut_point,
                  float lambda);
  vector<float> get_scores(const vector<float> &feature,
                           const Matrix &gradients,
                           const vector<int> &task,
                           const int &task_num,
                           const vector<int> &sample_index,
                           float cut_point, float lambda);

};

#endif //MTREE_MULTITASKUPDATER_H
