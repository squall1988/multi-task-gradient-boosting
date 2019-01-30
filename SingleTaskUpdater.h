//
// Created by squall on 18-6-11.
//

#ifndef MTREE_SINGLETASKOBJ_H
#define MTREE_SINGLETASKOBJ_H

#include "Updater.h"

class SingleTaskUpdater : public Updater {
 public:
  SingleTaskUpdater() {}
  ~SingleTaskUpdater() {}
  float get_score(const vector<float> &feature,
                  const Matrix &gradients,
                  const vector<int> &sample_index,
                  float cut_point, float lambda);

  vector<float> get_scores(const vector<float> &feature,
                           const Matrix &gradients,
                           const vector<int> &task,
                           const int &task_num,
                           const vector<int> &sample_index,
                           float cut_point, float lambda);

};

#endif //MTREE_SINGLETASKOBJ_H
