//
// Created by squall on 18-6-11.
//

#ifndef MTREE_OBJECTIVE_H
#define MTREE_OBJECTIVE_H

#include <vector>
#include "utils.h"

using namespace std;
using namespace common;

class Updater {
 public:
  Updater() {}
//  virtual float get_gradient() = 0;
  virtual ~Updater() {}
  virtual float get_score(const vector<float> &feature,
                          const Matrix &gradients,
                          const vector<int> &sample_index,
                          float cut_point,
                          float lambda) = 0;
  virtual vector<float> get_scores(const vector<float> &feature,
                                   const Matrix &gradients,
                                   const vector<int> &task,
                                   const int &task_num,
                                   const vector<int> &sample_index,
                                   float cut_point, float lambda) = 0;
};

#endif //MTREE_OBJECTIVE_H
