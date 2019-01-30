//
// Created by squall on 18-6-11.
//

#ifndef MTREE_TREE_H
#define MTREE_TREE_H

#include "Node.h"
#include "LinearLoss.h"

class Tree {
 public:
  Tree(int max_depth, float lambda,
       float beta,
       int feature_size,
       int min_sample_num,
       float learning_rate,
       string regularization,
       Loss *loss,
       Updater *objective) : max_depth(max_depth),
                             lambda(lambda),
                             beta(beta),
                             feature_size(feature_size),
                             min_sample_num(min_sample_num),
                             learning_rate(learning_rate),
                             regularization(regularization) {
    root = NULL;
    this->loss = loss;
    this->objective = objective;

  }
  int train(Dataset const &dataset);
  int predict(const Dataset &dataset, vector<float> &pred);
 private:
  Node *root;
  float lambda;
  float beta; // regularization coefficient
  int max_depth;
  Loss *loss;
  int feature_size;
  Updater *objective;
  int min_sample_num;
  float learning_rate;
  string regularization;
};
#endif //MTREE_TREE_H
