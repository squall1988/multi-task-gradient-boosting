//
// Created by squall on 18-6-11.
//

#ifndef MTREE_LOSS_H
#define MTREE_LOSS_H

#include <vector>
#include "utils.h"
using namespace std;
using namespace common;

class Loss {
 public:
  virtual ~Loss() {}
  virtual int get_gradient(const vector<float> &pred,
                           const vector<int> &label,
                           Matrix &gradients) = 0;

};

#endif //MTREE_LOSS_H
