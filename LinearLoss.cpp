//
// Created by squall on 18-6-11.
//

#include "LinearLoss.h"
#include "error.h"
#include <iostream>

int LinearLoss::get_gradient(const vector<float> &pred, const vector<int> &label, Matrix &gradients) {
  if (pred.size() != label.size()) {
    return LOSS_ERROR;
  }
  for (int i = 0; i < pred.size(); ++i) {
    float g = pred[i] - label[i];
    float h = 1.0f;
    vector<float> tmp{g, h};
//    tmp.push_back(g);
//    tmp.push_back(h);
    gradients.push_back(tmp);
  }
  return SUCCESS;
}

int LogisticLoss::get_gradient(const vector<float> &pred, const vector<int> &label, Matrix &gradients) {
  if (pred.size() != label.size()) {
    return LOSS_ERROR;
  }
  for (int i = 0; i < pred.size(); ++i) {
    const float z = 1.0f / (1.0f + std::exp(-pred[i]));
    float g = z - label[i];
    float h = z * (1.0f - z);
//    std::cout<<g<<" "<<h<<endl;
    vector<float> tmp{g, h};
    gradients.push_back(tmp);
  }
  return SUCCESS;
}