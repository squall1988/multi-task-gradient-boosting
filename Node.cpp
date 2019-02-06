//
// Created by squall on 18-6-11.
//

#include <iostream>
#include <set>
#include "Node.h"
#include<numeric>
#include <chrono>

int Node::calc_node_score(const Matrix &gradients, float lambda) {
  float sum_g = 0.0f;
  float sum_h = 0.0f;
  for (int i = 0; i < this->sample_index.size(); ++i) {
    sum_g += gradients[this->sample_index[i]][0];
    sum_h += gradients[this->sample_index[i]][1];
  }
  this->node_score = (sum_g * sum_g) / (sum_h + lambda);
  return SUCCESS;
}

// calculate structure score of each task
int Node::calc_node_scores(Dataset const &data, float lambda) {
  vector<vector<float>> used_data;
  vector<float> used_label;
  vector<int> used_task;
  Matrix node_gradients;
  data.get_sample_by_index(this->sample_index, used_data, used_label, used_task, node_gradients);
  vector<float> sum_g(data.get_task_num() + 1, 0.0f);
  vector<float> sum_h(data.get_task_num() + 1, 0.0f);
  for (int i = 0; i < this->sample_index.size(); ++i) {
    sum_g[used_task[i]] += node_gradients[i][0];
    sum_h[used_task[i]] += node_gradients[i][1];
  }
  this->node_scores.push_back(0.0f);
  for (int i = 1; i <= data.get_task_num(); ++i) {
    this->node_scores.push_back((sum_g[i] * sum_g[i]) / (sum_h[i] + lambda));
  }
  return SUCCESS;
}

int Node::calc_node_weight(const Matrix &gradients, float lambda) {
  float sum_g = 0.0f;
  float sum_h = 0.0f;
  for (int i = 0; i < this->sample_index.size(); ++i) {
    sum_g += gradients[this->sample_index[i]][0];
    sum_h += gradients[this->sample_index[i]][1];
  }
//  cout<<"This is : " << sum_g<<", this is sum_h: "<<sum_h<<endl;
  this->weight = -sum_g / (sum_h + lambda);
//  cout<<"leaf weight: "<<this->weight<<endl;
  return SUCCESS;
};

int Node::find_split_point(Dataset const &data, float lambda) {
  vector<vector<float>> used_data;
  vector<float> used_label;
  vector<int> used_task;
  Matrix node_gradients;
  data.get_sample_by_index(this->sample_index, used_data, used_label, used_task, node_gradients);
  int best_feature = -1;
  float best_cut_point = -1.0f;
  float best_score = -1.0f;
//  std::cout<<this->node_score<<endl;
  for (int i = 0; i < data.get_common_feature_size(); ++i) {
    float cut_point, score;
    ERROR_CODE_CHECK(this->find_split_point_single_feature(used_data[i],
                                                           used_label,
                                                           node_gradients,
                                                           cut_point,
                                                           score,
                                                           lambda));
    if (score > best_score) {
      best_score = score;
      best_cut_point = cut_point;
      best_feature = i;
    }
  }
//  cout<<best_score<<endl;
  this->feature_index = best_feature;
  this->split_point = best_cut_point;
//  this->node_score = best_score;
  if (this->feature_index == -1) {
    this->right = NULL;
    this->left = NULL;
  } else {
    // generate the corresponding child node.
    this->generate_node(used_data[best_feature], best_cut_point);
  }
  return 0;
}

int Node::find_split_point_single_feature(const vector<float> &feature,
                                          const vector<float> &label,
                                          const Matrix &gradients,
                                          float &cut_point,
                                          float &score,
                                          float lambda) {
  if (feature.empty()) {
    return NODE_SAMPLE_EMPTY;
  }
  set<float> unique_num;
  for (int i = 0; i < feature.size(); i++) {
    unique_num.insert(feature[i]);
  }
  set<float> candidate_cut_points;
  if (unique_num.size() > 100) {
    this->find_candidate_split_feature_value(feature, gradients, candidate_cut_points);
  } else {
    candidate_cut_points = unique_num;
  }
//  std::cout<<unique_num.size()<<" "<<feature.size()<<endl;
  float best_score = -1.0f;
  float best_cut_point = 0;
  for (set<float>::iterator it = candidate_cut_points.begin(); it != candidate_cut_points.end(); ++it) {
    float tmp = this->score_obj->get_score(feature, gradients,
                                           this->sample_index,
                                           *it, lambda
    );
    // current cut point score.
    tmp = tmp - this->node_score;
    if (tmp > best_score) {
      best_score = tmp;
      best_cut_point = *it;
    }
  }
//  std::cout<<best_score<<endl;
  score = best_score;
  cut_point = best_cut_point;
  return 0;
}

int Node::find_split_point_common(Dataset const &data, float lambda, float beta, string regularization) {
  vector<vector<float>> used_data;
  vector<float> used_label;
  vector<int> used_task;
  Matrix node_gradients;
  data.get_sample_by_index(this->sample_index, used_data, used_label, used_task, node_gradients);
  vector<int> used_data_sizes(data.get_task_num() + 1, 0);
  for (int i = 0; i < used_task.size(); ++i) {
    used_data_sizes[used_task[i]] += 1;
  }
  int best_feature = -1;
  float best_cut_point = 0.0f;
  float best_score = -1000.0f;
//  std::cout<<this->node_score<<endl;
  for (int i = 0; i < data.get_common_feature_size(); ++i) {
    float cut_point, score;
    ERROR_CODE_CHECK(this->find_split_point_single_feature_common(used_data[i],
                                                                  used_label,
                                                                  used_task,
                                                                  used_data_sizes,
                                                                  data.get_task_num(),
                                                                  node_gradients,
                                                                  cut_point,
                                                                  score,
                                                                  lambda,
                                                                  beta,
                                                                  regularization));
    if (score > best_score) {
      best_score = score;
      best_cut_point = cut_point;
      best_feature = i;
    }
  }

//  cout<<best_feature<<" "<<best_score<<" "<<best_cut_point<<endl;
  this->feature_index = best_feature;
  this->split_point = best_cut_point;
//  this->node_score = best_score;

  if (this->feature_index == -1) {
    this->right = NULL;
    this->left = NULL;
  } else {
    // generate the corresponding child node.
    this->generate_node(used_data[best_feature], best_cut_point);
  }
  return 0;
}

int Node::find_split_point_single_feature_common(const vector<float> &feature,
                                                 const vector<float> &label,
                                                 const vector<int> &task,
                                                 const vector<int> &data_sizes,
                                                 const int &task_num,
                                                 const Matrix &gradients,
                                                 float &cut_point,
                                                 float &score,
                                                 float lambda,
                                                 float beta,
                                                 string regularization) {
  if (feature.empty()) {
    return NODE_SAMPLE_EMPTY;
  }
  set<float> unique_num;
  for (int i = 0; i < feature.size(); i++) {
    unique_num.insert(feature[i]);
  }
  set<float> candidate_cut_points;
  cout << " outof the count " << unique_num.size() << endl;
  if (unique_num.size() > 100) {
    this->find_candidate_split_feature_value(feature, gradients, candidate_cut_points);
  } else {
    candidate_cut_points = unique_num;
  }

  cout << unique_num.size() << endl;
  float best_score = -1000.0f;
  float best_cut_point = 0;
  for (set<float>::iterator it = candidate_cut_points.begin(); it != candidate_cut_points.end(); ++it) {
    vector<float> tmp_scores = this->score_obj->get_scores(feature, gradients, task, task_num,
                                                           this->sample_index,
                                                           *it, lambda
    );
    float tmp_score = this->score_obj->get_score(feature, gradients, this->sample_index, *it, lambda);
    float raw_gain_score = (tmp_score - this->node_score);
    if (raw_gain_score <= 0.0f) {
      continue;
    }
    // gain rate for each task
    vector<float> tmp_gains(task_num, 0.0f);
    for (int i = 1; i <= task_num; ++i) {
      //tmp_gains[i] can be equal to 0
      tmp_gains[i - 1] = tmp_scores[i] - this->node_scores[i];
    }
    // current cut point score.
    float reg = 0.0f;
    float gain_score = 0.0f;
    if (regularization == "variance") {
      stddev_regularization(tmp_gains, reg);
      gain_score = raw_gain_score - beta * reg;
    } else if (regularization == "entropy") {
      entropy_regularization(tmp_gains, reg);
      gain_score = reg * raw_gain_score;
    } else if (regularization == "weight_entropy") {
      entropy_regularization(tmp_gains, reg);
      gain_score = (1 + beta * reg) * gain_score;
    } else {
      gain_score = raw_gain_score;
    }
    if (gain_score > best_score) {
      best_score = gain_score;
      best_cut_point = *it;
    }
  }
//  std::cout<<best_score<<endl;
  score = best_score;
  cut_point = best_cut_point;
  return 0;
}

int Node::find_split_point_thread(Dataset const &data, float lambda, int feature_size) {
  vector<vector<float>> used_data;
  vector<float> used_label;
  vector<int> used_task;
  Matrix node_gradients;
  data.get_sample_by_index(this->sample_index, used_data, used_label, used_task, node_gradients);
  int best_feature = -1;
  float best_cut_point = -1.0f;
  float best_score = -1.0f;
  ThreadPool pool(THREAD_NUM);
  std::vector<std::future<int> > results;
  auto *cut_point = new float[feature_size]();
  auto *score = new float[feature_size]();
  for (int i = 0; i < feature_size; ++i) {
    results.emplace_back(pool.enqueue(Node::find_split_point_single_feature_static,
                                      used_data[i],
                                      used_label,
                                      node_gradients,
                                      this->sample_index,
                                      this->score_obj,
                                      this->node_score,
                                      lambda,
                                      i,
                                      cut_point,
                                      score));
  }
  for (auto &&x: results) {
    if (x.get() != SUCCESS) {
      return NODE_SPLIT_ERROR;
    }
  }
  for (int i = 0; i < feature_size; ++i) {
    if (score[i] > best_score) {
      best_score = score[i];
      best_feature = i;
      best_cut_point = cut_point[i];
    }
  }
  delete[] score;
  delete[] cut_point;
//  cout<<best_score<<endl;
  this->feature_index = best_feature;
  this->split_point = best_cut_point;
//  this->node_score = best_score;
  if (this->feature_index == -1) {
    this->right = NULL;
    this->left = NULL;
  } else {
    // generate the corresponding child node.
    this->generate_node(used_data[best_feature], best_cut_point);
  }
  return SUCCESS;
}

int Node::find_split_point_single_feature_static(const vector<float> &feature,
                                                 const vector<float> &label,
                                                 const Matrix &gradients,
                                                 const vector<int> &sample_index,
                                                 Updater *score_obj,
                                                 float node_score,
                                                 float lambda,
                                                 int feature_index,
                                                 float *cut_point,
                                                 float *score) {
  if (feature.empty()) {
    return NODE_SAMPLE_EMPTY;
  }
  set<float> unique_num;
  for (int i = 0; i < feature.size(); i++) {
    unique_num.insert(feature[i]);
  }
  set<float> candidate_cut_points;
  if (unique_num.size() > 100) {
    find_candidate_split_feature_value(feature, gradients, candidate_cut_points);
  } else {
    candidate_cut_points = unique_num;
  }
//  std::cout<<unique_num.size()<<" "<<feature.size()<<endl;
  float best_score = -1.0f;
  float best_cut_point = 0;
  for (set<float>::iterator it = candidate_cut_points.begin(); it != candidate_cut_points.end(); ++it) {
    float tmp = score_obj->get_score(feature, gradients,
                                     sample_index,
                                     *it, lambda
    );
    // current cut point score.
    tmp = tmp - node_score;
    if (tmp > best_score) {
      best_score = tmp;
      best_cut_point = *it;
    }
  }
//  std::cout<<best_score<<endl;
  score[feature_index] = best_score;
  cut_point[feature_index] = best_cut_point;
  return SUCCESS;
}

int Node::find_split_point_common_thread(
    Dataset const &data, float lambda, float beta, string regularization, int feature_size) {
  vector<vector<float>> used_data;
  vector<float> used_label;
  vector<int> used_task;
  Matrix node_gradients;
  data.get_sample_by_index(this->sample_index, used_data, used_label, used_task, node_gradients);
  vector<int> used_data_sizes(data.get_task_num() + 1, 0);
  for (int i = 0; i < used_task.size(); ++i) {
    used_data_sizes[used_task[i]] += 1;
  }
  int best_feature = -1;
  float best_cut_point = -1.0f;
  float best_score = -1000.0f;
  ThreadPool pool(THREAD_NUM);
  std::vector<std::future<int> > results;
  auto *cut_point = new float[feature_size]();
  auto *score = new float[feature_size]();
  for (int i = 0; i < feature_size; ++i) {
    results.emplace_back(pool.enqueue(Node::find_split_point_single_feature_common_static,
                                      used_data[i],
                                      used_label,
                                      used_task,
                                      used_data_sizes,
                                      data.get_task_num(),
                                      node_gradients,
                                      this->sample_index,
                                      this->score_obj,
                                      this->node_score,
                                      this->node_scores,
                                      lambda,
                                      beta,
                                      i,
                                      cut_point,
                                      score,
                                      regularization));
  }
  for (auto &&x: results) {
    if (x.get() != SUCCESS) {
      return NODE_SPLIT_ERROR;
    }
  }
  for (int i = 0; i < feature_size; ++i) {
    if (score[i] > best_score) {
      best_score = score[i];
      best_feature = i;
      best_cut_point = cut_point[i];
    }
  }
  delete[] score;
  delete[] cut_point;
  this->feature_index = best_feature;
  this->split_point = best_cut_point;
//  this->node_score = best_score;

  if (this->feature_index == -1) {
    this->right = NULL;
    this->left = NULL;
  } else {
//    cout<<"best_cut_point: "<<best_cut_point<<endl;
    // generate the corresponding child node.
    this->generate_node(used_data[best_feature], best_cut_point);
  }
  return SUCCESS;
}

int Node::find_split_point_single_feature_common_static(const vector<float> &feature,
                                                        const vector<float> &label,
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
                                                        float *score,
                                                        string regularization) {
  if (feature.empty()) {
    return NODE_SAMPLE_EMPTY;
  }
  set<float> unique_num;
  for (int i = 0; i < feature.size(); i++) {
    unique_num.insert(feature[i]);
  }
  set<float> candidate_cut_points;
  if (unique_num.size() > 100) {
    find_candidate_split_feature_value(feature, gradients, candidate_cut_points);
  } else {
    candidate_cut_points = unique_num;
  }


  float best_score = -1000.0f;
  float best_cut_point = 0;
  for (set<float>::iterator it = candidate_cut_points.begin(); it != candidate_cut_points.end(); ++it) {
    vector<float> tmp_scores = score_obj->get_scores(feature, gradients, task, task_num,
                                                     sample_index,
                                                     *it, lambda
    );
    float tmp_score = score_obj->get_score(feature, gradients, sample_index, *it, lambda);
    float raw_gain_score = (tmp_score - node_score);
    if (raw_gain_score <= -1.0f) {
      continue;
    }
    // gain for each task
    vector<float> tmp_gains(task_num, 0.0f);
    for (int i = 1; i <= task_num; ++i) {
      tmp_gains[i - 1] = tmp_scores[i] - node_scores[i];
    }
    // current cut point score.

    float reg = 0.0f;
    float gain_score = 0.0f;
    if (regularization == "variance") {
      stddev_regularization(tmp_gains, reg);
      gain_score = raw_gain_score - beta * reg;
    } else if (regularization == "entropy") {
      entropy_regularization(tmp_gains, reg);
      gain_score = reg * raw_gain_score;
    } else if (regularization == "weight_entropy") {
      entropy_regularization(tmp_gains, reg);
      gain_score = (1 + beta * reg) * gain_score;
    } else {
      gain_score = raw_gain_score;
    }
    if (gain_score > best_score) {
      best_score = gain_score;
      best_cut_point = *it;
    }
  }
//  std::cout<<best_score<<endl;
  score[feature_index] = best_score;
  cut_point[feature_index] = best_cut_point;
  return SUCCESS;
}

int Node::stddev_regularization(const vector<float> &task_gains, float &reg) {
  // calculate mean and variance
  float sum = std::accumulate(std::begin(task_gains), std::end(task_gains), 0.0f);
  float mean = sum / task_gains.size();
  float variance = 0.0f;
  for (vector<float>::const_iterator it = task_gains.begin(); it != task_gains.end(); ++it) {
    variance += (*it - mean) * (*it - mean);
  }
  reg = variance / (task_gains.size() - 1);
  return SUCCESS;
}

int Node::entropy_regularization(const vector<float> &task_gains, float &reg) {
  float z = 0.0f;
  vector<float> c;
  float gain_sum = 0.0f;
  for (vector<float>::const_iterator it = task_gains.begin(); it != task_gains.end(); ++it) {
    float task_gain = *it <= 0 ? 0 : *it;
    gain_sum += task_gain;
    c.push_back(task_gain);
  }
  for (vector<float>::const_iterator it = c.begin(); it != c.end(); ++it) {
    if (*it == 0) continue;
    z -= (*it / gain_sum) * std::log(*it / gain_sum);
  }
  reg = z;
  return SUCCESS;
}


int Node::generate_node(const vector<float> &feature, float cut_points) {

  vector<int> left_node_sample;
  vector<int> right_node_sample;
  if (feature.size() != this->sample_index.size() || feature.size() == 0) {
    return NODE_SAMPLE_EMPTY;
  }
  for (int i = 0; i < feature.size(); ++i) {
    int index = this->sample_index[i];
    feature[i] >= cut_points ? right_node_sample.push_back(index) : left_node_sample.push_back(index);
  }
  Node *left_node = NULL;
  Node *right_node = NULL;
  if (left_node_sample.size() != 0 && right_node_sample.size() != 0) {
    // 没问题，只有当两个叶子结点都样本的时候才能进行分裂，否则就不分裂了。
    left_node = new Node(left_node_sample, this, this->score_obj, this->min_sample_num);
    right_node = new Node(right_node_sample, this, this->score_obj, this->min_sample_num);
  }
//  cout<<left_node_sample.size()<< " " <<right_node_sample.size()<<endl;
  this->right = right_node;
  this->left = left_node;
  return SUCCESS;
}

int Node::get_sample_size() const {
  return this->sample_index.size();
}

int Node::find_candidate_split_feature_value(const vector<float> &feature,
                                             const Matrix &gradients,
                                             set<float> &candidate_cut_points) {
  vector<pair<float, float>> d;
  for (int i = 0; i < feature.size(); ++i) {
    d.emplace_back(make_pair(feature[i], gradients[i][1]));
  }
  // sort by feature value
  sort(d.begin(), d.end(), cmp);
  float sum_h = 0.0f;
  for (int i = 0; i < d.size(); ++i) {
    sum_h += d[i].second;
  }
  // init candidate_cut_points
  candidate_cut_points.insert(d[0].first);
  // 上一个特征值的rank
  float pre_rank = 0.0f;
  // 当前二阶导数和
  float cur_sum_h = d[0].second;
  // 当前符合x<z的二阶导数和
  float cur_real_sum_h = 0.0f;
  float pre_feature_value = d[0].first;
  for (int i = 1; i < d.size(); ++i) {
    if (d[i].first > pre_feature_value) {
      cur_real_sum_h = cur_sum_h;
    }
    float cur_rank = cur_real_sum_h / sum_h;
    if (cur_rank - pre_rank >= 0.01) {
      candidate_cut_points.insert(d[i].first);
      pre_rank = cur_rank;
    }
    pre_feature_value = d[i].first;
    cur_sum_h += d[i].second;
  }
  return 0;
}