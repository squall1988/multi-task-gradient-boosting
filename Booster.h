//
// Created by squall on 18-6-12.
//

#ifndef MTREE_BOOSTER_H
#define MTREE_BOOSTER_H

#include "tree.h"
#include <ctime>
template<typename LOSS, typename UPDATER>
class Booster {
 public:
  Booster(int max_num_round,
          int common_num_round,
          int max_depth,
          float lambda,
          float beta,
          int min_sample_leaf,
          float learning_rate,
          string regularization) {
    this->max_num_round = max_num_round;
    this->common_num_round = common_num_round;
    this->max_depth = max_depth;
    this->lambda = lambda;
    this->beta = beta;
    this->min_sample_leaf = min_sample_leaf;
    this->learning_rate = learning_rate;
    this->regularization = regularization;
  }

  int train(Dataset &dataset,
            Dataset &eval_set,
            string eval_metric,
            int early_stopping_rounds,
            bool verbose);

  int predict(Dataset &dataset, vector<float> &score, const string &log_path);
  // 对每个task单独预测
  int single_predict(const Dataset &dataset, vector<float> &pred, const int &task_id, float &loss_score);
  // calculate loss score
  int calculate_loss_score(const vector<int> &label,
                           const vector<float> &pred,
                           const string &eval_metric,
                           const int &task_id,
                           float &loss_score);
 private:
  int max_num_round;
  int common_num_round;
  int max_depth;
  float lambda;
  float beta;
  int min_sample_leaf;
  float learning_rate;
  string regularization;
  vector<int> single_num_rounds;
  string eval_metric;
  int early_stopping_rounds;
  bool verbose;
  vector<Tree *> common_trees;
  vector<vector<Tree *>> single_trees;
  vector<pair<int, float> > commmon_best_iterations;
  vector<pair<int, float> > single_best_iterations;
};

#endif //MTREE_BOOSTER_H
