//
// Created by squall on 18-6-11.
//

#ifndef MTREE_LINEARLOSS_H
#define MTREE_LINEARLOSS_H

#include "Loss.h"
#include<cmath>

class LinearLoss : public Loss {
 public:
   int get_gradient(const vector<float> &pred, const vector<int> &label, Matrix &gradients) override;
};

class LogisticLoss : public Loss {
 public:
  int get_gradient(const vector<float> &pred, const vector<int> &label, Matrix &gradients) override;
};
#endif //MTREE_LINEARLOSS_H
