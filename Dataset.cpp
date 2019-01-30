//
// Created by squall on 18-6-11.
//

#include <assert.h>
#include <iostream>
#include "Dataset.h"


int Dataset::load_data_from_file(const string &file_name, const char *delimiter) {
  ifstream in_file(file_name);
  string line;
  int data_size = 0;
  while (in_file >> line) {
    data_size += 1;
    // split the line and transform the data to float, insert into the data.
    vector<string> split_result = common::Split(line, *delimiter);
//    std::cout<<split_result.size()<<" "<<this->feature_size<<endl;
    assert(split_result.size() == this->feature_size + 2);
    for (int i = 0; i < this->feature_size; i++) {
      float tmp = (float) atof(split_result[i].c_str());
      this->data[i].push_back(tmp);
    }
    // store the label and the task;
    this->label.push_back(atoi(split_result[this->feature_size].c_str()));
    this->task.push_back(atoi(split_result[this->feature_size + 1].c_str()));
  }

#ifdef DEBUG
  for (int i = 0; i < label.size(); ++i) {
    cout << label[i] << endl;
  }
#endif

  this->dataset_size = data_size;
  return 0;
}

int Dataset::get_sample_by_index(vector<int> &index,
                                 vector<vector<float>> &selected_sample,
                                 vector<int> &selected_label,
                                 vector<int> &selected_task,
                                 Matrix &selected_gradients) const {
  for (int i = 0; i < this->feature_size; i++) {
    vector<float> tmp;
    for (int j = 0; j < index.size(); j++) {
      tmp.push_back(this->data[i][index[j]]);
    }
    selected_sample.push_back(tmp);
  }
  for (int i = 0; i < index.size(); i++) {
//    cout<<this->label.size()<<" "<<this->task.size()<<" "<<this->gradients.size()<<endl;
    selected_label.push_back(this->label[index[i]]);
    selected_task.push_back(this->task[index[i]]);
    selected_gradients.push_back(this->gradients[index[i]]);
  }
  return 0;
}

int Dataset::get_data_by_tasks(vector<Dataset> &datasets) const {
  vector<Matrix> data(this->task_num + 1);
  vector<vector<int>> labels(this->task_num + 1);
  vector<vector<int>> tasks(this->task_num + 1);
  vector<Matrix> gradients(this->task_num + 1);
  vector<Matrix> tmp_data(this->task_num + 1);
//  cout<<this->dataset_size<<endl;
  // 存储方式：一行一个sample
  for (int i = 0; i < this->dataset_size; ++i) {
    vector<float> tmp;
    for (int j = 0; j < this->feature_size; ++j) {
      tmp.push_back(this->data[j][i]);
    }
    tmp_data[this->task[i]].push_back(tmp);
  }
  // 存储方式改为一行一个feature
  for (int i = 1; i <= this->task_num; ++i) {
    for (int j = 0; j < this->feature_size; ++j) {
      vector<float> tmp;
      for (int k = 0; k < tmp_data[i].size(); ++k) {
        tmp.push_back(tmp_data[i][k][j]);
      }
      data[i].push_back(tmp);
    }
  }
  for (int i = 0; i < this->dataset_size; ++i) {
    labels[this->task[i]].push_back(this->label[i]);
    tasks[this->task[i]].push_back(this->task[i]);
    if (!this->gradients.empty()) {
      gradients[this->task[i]].push_back(this->gradients[i]);
    }
  }
  for (int i = 0; i < this->task_num; ++i) {
    datasets[i].data = data[i + 1];
    datasets[i].label = labels[i + 1];
    datasets[i].task_num = 1;
    if (!this->gradients.empty()) {
      datasets[i].gradients = gradients[i + 1];
    }
    datasets[i].task = tasks[i + 1];
    datasets[i].dataset_size = tmp_data[i + 1].size();
//    cout<<datasets[i].dataset_size<<endl;
  }
  return 0;
}

int Dataset::get_data_by_index(vector<int> &index, Dataset &dataset) const {
  vector<vector<float>> used_data;
  vector<int> used_label;
  vector<int> used_task;
  for (int i = 0; i < this->feature_size; i++) {
    vector<float> tmp;
    for (int j = 0; j < index.size(); j++) {
      tmp.push_back(this->data[i][index[j]]);
    }
    used_data.push_back(tmp);
  }
  for (int i = 0; i < index.size(); i++) {
    used_label.push_back(this->label[index[i]]);
    used_task.push_back(this->task[index[i]]);
  }
  dataset.data = used_data;
  dataset.label = used_label;
  dataset.task = used_task;
  dataset.dataset_size = used_data[0].size();
  return 0;
}

vector<pair<Dataset, Dataset>> Dataset::shuffle_split(const int &n_splits,
                                                      const float &test_size,
                                                      const int &random_state) const {
  vector<pair<Dataset, Dataset>> datasets;
  int n = this->get_data_size();
  int m = (int) ((float) (n) * test_size);
  common::Random seed(random_state);
  for (int i = 0; i < n_splits; ++i) {
    pair<Dataset, Dataset> p = train_test_split(test_size, seed.NextInt(1, 100));
    datasets.push_back(p);
  }
  return datasets;
}

pair<Dataset, Dataset> Dataset::train_test_split(const float &test_size, const int &random_state) const {
  int n = this->get_data_size();
  int m = (int) ((float) (n) * test_size);
  common::Random random(random_state);

  vector<int> test_index;
  vector<int> train_index;
  test_index = random.Sample(n, m);
  int j = 0;
  for (int k = 0; k < n; ++k) {
    if (test_index[j] != k) {
      train_index.push_back(k);
    } else {
      ++j;
    }
  }

  Dataset train(this->get_feature_size());
  Dataset test(this->get_feature_size());
  train.set_task_num(this->get_task_num());
  test.set_task_num(this->get_task_num());
  this->get_data_by_index(train_index, train);
  this->get_data_by_index(test_index, test);
  return make_pair(train, test);
};