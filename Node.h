//
// Created by squall on 18-6-11.
//

#ifndef MTREE_NODE_H
#define MTREE_NODE_H
#include <vector>
#include "Dataset.h"
#include "error.h"
#include "SingleTaskUpdater.h"
#include "ThreadPool.h"

using namespace std;

class Node {
 public:
  Node(vector<int> &feature_index, Node *father_node, Updater *obj, int min_sample_num, float node_score) {
//    this->feature_index = feature_index.get();
    this->father_node = father_node;
    this->sample_index = feature_index;
    this->score_obj = obj;
    this->min_sample_num = min_sample_num;
    this->node_score = node_score;
  }

  Node(vector<int> &feature_index, Node *father_node, Updater *obj, int min_sample_num) {
//    this->feature_index = feature_index.get();
    this->father_node = father_node;
    this->sample_index = feature_index;
    this->score_obj = obj;
    this->min_sample_num = min_sample_num;
  }
  /*! \belief: find the best split point of the Dataset base on current sample index.
            generate coresponding child Node.
   */
  int find_split_point(Dataset const &data, float lambda);
  /*! \belief: find best split point for each feature. */
  int find_split_point_single_feature(const vector<float> &feature,
                                      const vector<int> &label,
                                      const Matrix &gradients,
                                      float &cut_point,
                                      float &score,
                                      float lambda);

  int find_split_point_common(Dataset const &data, float lambda, float beta, string regularization);
  /*! \belief: find best split point for each feature. */
  int find_split_point_single_feature_common(const vector<float> &feature,
                                             const vector<int> &label,
                                             const vector<int> &task,
                                             const vector<int> &data_sizes,
                                             const int &task_num,
                                             const Matrix &gradients,
                                             float &cut_point,
                                             float &score,
                                             float lambda,
                                             float beta,
                                             string regularization);

  static int find_split_point_single_feature_static(const vector<float> &feature,
                                                    const vector<int> &label,
                                                    const Matrix &gradients,
                                                    const vector<int> &sample_index,
                                                    Updater *score_obj,
                                                    float node_score,
                                                    float lambda,
                                                    int feature_index,
                                                    float *cut_point,
                                                    float *score
  );

  /*! \belif: single train: find split point, multi thread version */
  int find_split_point_thread(Dataset const &data, float lambda);

  static int find_split_point_single_feature_common_static(const vector<float> &feature,
                                                           const vector<int> &label,
                                                           const vector<int> &task,
                                                           const vector<int> &data_sizes,
                                                           const int &task_num,
                                                           const Matrix &gradients,
                                                           const vector<int> &sample_index,
                                                           Updater *score_obj,
                                                           const float &node_score,
                                                           const vector<float> &node_scores,
                                                           const float &lambda,
                                                           const float &beta,
                                                           const int &feature_index,
                                                           float *cut_point,
                                                           float *score

  );

  /*! \belif: common train: find split point, multi thread version */
  int find_split_point_common_thread(Dataset const &data, float lambda, float beta);

  /*! \belief: stddev-based regularization. */
  static int stddev_regularization(const vector<float> &task_gains, float &reg);

  /*! \belief: entropy-based regularization. */
  static int entropy_regularization(const vector<float> &task_gains, float &reg);

  /*! \belief: find candidate split cut point by using weighted quantile sketch */
  int find_candidate_split_feature_value(const vector<float> &feature,
                                         const Matrix &gradients,
                                         set<float> &candidate_cut_points);

  int calc_node_weight(const Matrix &gradients, float lambda);

  int calc_node_score(const Matrix &gradients, float lambda);

  int calc_node_scores(Dataset const &data, float lambda);

  /*! @belief: generate child Node. */
  int generate_node(const vector<float> &feature, float cut_points);

  int get_sample_size() const;
  Node *get_left_node() {
    return this->left;
  }
  Node *get_right_node() {
    return this->right;
  }
  void set_is_leaf(bool is_leaf) {
    this->is_leaf = is_leaf;
  }

  int get_feature_index() const {
    return this->feature_index;
  }

  float get_cut_point() const {
    return this->split_point;
  }

  float get_node_score() const {
    return this->node_score;
  }

  vector<float> get_node_scores() const {
    return this->node_scores;
  }

  bool get_is_leaf() const {
    return this->is_leaf;
  }

  inline void set_weight(float weight) {
    this->weight = weight;
  }

  inline float get_weight() const {
    return this->weight;
  }

 private:
  // store the sample index for this Node.
  vector<int> sample_index;
  // store the split point.
  float split_point;
  //  store the feature num.
  int feature_index;
  //  if this is a leaf Node.
  bool is_leaf;
  Node *left = NULL;
  Node *right = NULL;
  /*! \belief: score function; */
  Updater *score_obj;
  /*! \belief: gain score of this node. */
  float node_score;
  /*! \belief: each task's gain score of this node. */
  vector<float> node_scores;

  Node *father_node;
  /*! \belief: min sample number in a leaf */
  int min_sample_num;
  /*! \belif: value of this node. */
  float weight;

};

#endif //MTREE_NODE_H
