//
// Created by squall on 18-6-11.
//

#ifndef MTREE_DATASET_H
#define MTREE_DATASET_H

#include <vector>
#include <string>
#include <fstream>
// #include <bits/unique_ptr.h>
#include <algorithm>
#include <functional>
#include "utils.h"
#include "Random.h"

using namespace std;
using namespace common;

class Dataset {
 public:
  Dataset(int common_feature_size, int task_num, const vector<int> &single_feature_size) :
      common_feature_size(common_feature_size), task_num(task_num){

    this->single_feature_size = single_feature_size;
    this->max_size = 0;
    for(int i = 0; i < single_feature_size.size(); ++i) {
      if (single_feature_size[i] > this->max_size) {
        this->max_size = single_feature_size[i];
      }
    }
    for (int i = 0; i < this->max_size; i++) {
      vector<float> feature;
      this->data.push_back(feature);
    }
    this->candidate_cut_points.resize(this->max_size);
  }

  /*! \belief: get sample from the file */
  int load_data_from_file(const string &file_name, const char *delimiter);
  /*! \belief: get sample by index. */
  int get_sample_by_index(vector<int> &index, vector<vector<float>> &selected_sample,
                          vector<float> &selected_label, vector<int> &selected_task,
                          Matrix &selected_gradients) const;

  int get_data_by_index(vector<int> &index, Dataset &dataset) const;

  int get_sample_by_task(int task_id, vector<vector<float>> &selected_sample,
                         vector<float> &selected_label, vector<int> &selected_task,
                         Matrix &selected_gradients) const;

  /*! \belief: split data by task_id. */
  int get_data_by_tasks(vector<Dataset> &datasets) const;

  /*! \belief: shuffle split data into train and test set. */
  vector<pair<Dataset, Dataset>> shuffle_split(const int &n_splits,
                                               const float &test_size,
                                               const int &random_state) const;

  vector<pair<Dataset, Dataset>> shuffle_split_by_size(const int n_splits,
                        const int train_size,
                        const int test_size,
                        const int random_state) const;

  /*! \belief: split data into train and test set. */
  pair<Dataset, Dataset> train_test_split(const float &test_size, const int &random_state) const;

  int set_gradients(const Matrix &gradients) {
    this->gradients = gradients;
    return SUCCESS;
  }

  int set_task_num(const int task_num) {
    this->task_num = task_num;
    return SUCCESS;
  }

  const Matrix &get_gradients() const {
    return this->gradients;
  }

  int get_common_feature_size() const {
    return this->common_feature_size;
  }

  int get_task_num() const {
    return this->task_num;
  }
  const vector<float> &get_label_data() const {
    return this->label;
  }

  const vector<int>& get_single_feature_size() const{
    return this->single_feature_size;
  }

  int get_single_feature_size(int task_num) {
    if (task_num > this->task_num) {
      return -1;
    }
    return this->single_feature_size[task_num];
  }

  const vector<int> &get_task_data() {
    return this->task;
  }

  const int get_data_size() const {
    return this->dataset_size;
  }

  const Matrix &get_data() const {
    return this->data;
  }

  set<float>& get_unique_points(int feature_index);

 private:
  Matrix data;
  int common_feature_size;
  vector<int> single_feature_size;
  int dataset_size;
  int task_num;
  vector<float> label;
  vector<int> task;
  Matrix gradients;
  vector<set<float>> candidate_cut_points;
  int max_size;


};

#endif //MTREE_DATASET_H
