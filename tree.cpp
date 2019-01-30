//
// Created by squall on 18-6-11.
//

#include <queue>
#include <iostream>
#include "tree.h"

int Tree::train(Dataset const &dataset) {
  /*
   * 传到这里以后树的gradient应该已经被设定好了。
   */
  vector<int> sample_index;
  for (int i = 0; i < dataset.get_label_data().size(); ++i) {
    sample_index.push_back(i);
  }
  root = new Node(sample_index, NULL, this->objective, this->min_sample_num);
  root->calc_node_score(dataset.get_gradients(), lambda);
  if (dataset.get_task_num() > 1) {
    root->calc_node_scores(dataset, lambda);
    root->find_split_point_common(dataset, lambda, beta, this->regularization);
  } else {
    root->find_split_point(dataset, lambda);
  }
  root->set_is_leaf(false);
  queue<Node *> node_queue;
  if (root->get_left_node() != NULL)
    node_queue.push(root->get_left_node());
  if (root->get_right_node() != NULL)
    node_queue.push(root->get_right_node());
  for (int i = 1; i < this->max_depth; ++i) {
    int queue_size = node_queue.size();
    for (int j = 0; j < queue_size; j++) {
//      cout<<"max_depth "<<i<<" queue "<<j<<endl;
      Node *tmp = node_queue.front();
      if (tmp != NULL && tmp->get_sample_size() > min_sample_num) {
        tmp->calc_node_score(dataset.get_gradients(), lambda);
        // not null and the sample size is greater than min_sample_num
        if (dataset.get_task_num() > 1) {
          tmp->calc_node_scores(dataset, lambda);
          tmp->find_split_point_common(dataset, lambda, beta, this->regularization);
        } else {
          tmp->find_split_point(dataset, lambda);
        }
        if (tmp->get_right_node() == NULL && tmp->get_left_node() == NULL) {
          tmp->set_is_leaf(true);
          tmp->calc_node_weight(dataset.get_gradients(), lambda);
          if (this->learning_rate > 0) {
            tmp->set_weight(tmp->get_weight() * (this->learning_rate));
          }
        } else {
          node_queue.push(tmp->get_left_node());
          node_queue.push(tmp->get_right_node());
        }

      } else {
        if (tmp != NULL) {
          tmp->calc_node_score(dataset.get_gradients(), lambda);
          if (dataset.get_task_num() > 1) {
            tmp->calc_node_scores(dataset, lambda);
          }
          tmp->calc_node_weight(dataset.get_gradients(), lambda);
          if (this->learning_rate > 0) {
            tmp->set_weight(tmp->get_weight() * (this->learning_rate));
          }
          tmp->set_is_leaf(true);
        }
      }
      node_queue.pop();
    }
  }
  // calculate the rest of leaf node score.
  int node_size = node_queue.size();
  cout << "This is rest of node size " << node_size << endl;
  for (int i = 0; i < node_size; ++i) {
    Node *tmp = node_queue.front();
//    cout << "this is leaf sample size: " << tmp->get_sample_size() << endl;
    tmp->calc_node_score(dataset.get_gradients(), lambda);
//    cout << dataset.get_task_num() << endl;
    if (dataset.get_task_num() > 1) {
      tmp->calc_node_scores(dataset, lambda);
    }
    tmp->calc_node_weight(dataset.get_gradients(), lambda);
    tmp->set_is_leaf(true);
    if (this->learning_rate > 0) {
//      cout << "this is weight: " << tmp->get_weight() << endl;
      tmp->set_weight(tmp->get_weight() * (this->learning_rate));
    }
    node_queue.pop();
  }
  return TREE_TRAIN_ERROR;
}

int Tree::predict(const Dataset &dataset, vector<float> &pred) {
  if (dataset.get_data_size() == 0 || root == NULL) {
    return TREE_PREDICT_ERROR;
  }
  cout << "go into the predict" << endl;
  const Matrix &data = dataset.get_data();
  for (int i = 0; i < dataset.get_data_size(); ++i) {
    Node *current = root;
    while (!current->get_is_leaf()) {
      int index = current->get_feature_index();
      float cut_point = current->get_cut_point();
#ifdef DEBUG
      cout << "this is index of the feature:" << index << endl;
      cout << "this is cut_point:" << cut_point << endl;
      cout << "data is : " << data[index][i] << endl;
#endif
      if (data[index][i] >= cut_point) {
        current = current->get_right_node();
      } else {
        current = current->get_left_node();
      }
#ifdef DEBUG
      if (current == NULL) {
        cout << "this is break" << endl;
        break;

      }
#endif
    }
    float score = current->get_weight();
    pred.push_back(score);
  }
  return SUCCESS;
}



