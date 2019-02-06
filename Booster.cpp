//
// Created by squall on 18-6-12.
//

#include <iostream>
#include <string>
#include "Booster.h"
#include "LinearLoss.h"
//#define DEBUG

template<typename LOSS, typename UPDATER>
int Booster<LOSS, UPDATER>::train(Dataset &dataset,
                                  Dataset &eval_set,
                                  string eval_metric,
                                  int early_stopping_rounds,
                                  bool verbose) {
  this->eval_metric = eval_metric;
  this->early_stopping_rounds = early_stopping_rounds;
  this->verbose = verbose;
  LOSS *loss = new LOSS();
  UPDATER *objective = new UPDATER;
  Tree *tree_base =
      new Tree(max_depth,
               lambda,
               beta,
               dataset.get_feature_size(),
               min_sample_leaf,
               learning_rate,
               regularization,
               loss,
               objective);
  vector<float> pred(dataset.get_data_size(), 0.5);
  vector<float> eval_pred(eval_set.get_data_size(), 0.5);
  Matrix gradient;
  loss->get_gradient(pred, dataset.get_label_data(), gradient);
  dataset.set_gradients(gradient);
  if (common_num_round != 0) {
    cout << "this is the 1th round common train" << endl;
    tree_base->train(dataset);
    common_trees.push_back(tree_base);
  }
  for (int i = 0; i < dataset.get_task_num(); ++i) {
    this->single_num_rounds.push_back(max_num_round - common_num_round);
  }
  // common training
  // 每个task的label和pred分开存储
  vector<vector<float>> labels(dataset.get_task_num() + 1);
  vector<int> task = dataset.get_task_data();
  vector<float> label = dataset.get_label_data();

  vector<vector<float>> eval_labels(eval_set.get_task_num() + 1);
  vector<int> eval_task = eval_set.get_task_data();
  vector<float> eval_label = eval_set.get_label_data();

  // 每个task最优迭代次数的预测结果，当early stop后preds中对应task的值将不再更新
  vector<vector<float>> best_preds(dataset.get_task_num() + 1);
  vector<vector<float>> best_eval_preds(eval_set.get_task_num() + 1);
  // 之前最优的loss scores
  vector<float> common_min_loss_scores(eval_set.get_task_num() + 1, 1.0f);
  for (int i = 0; i < pred.size(); ++i) {
    best_preds[task[i]].push_back(pred[i]);
    labels[task[i]].push_back(label[i]);
  }
  for (int i = 0; i < eval_pred.size(); ++i) {
    best_eval_preds[eval_task[i]].push_back(eval_pred[i]);
    eval_labels[eval_task[i]].push_back(eval_label[i]);
  }
  //标志该task是否到了best iteration
  vector<bool> flag(dataset.get_task_num() + 1, false);
  // 记录该task有多少轮没有提升了
  vector<int> common_accum_rounds(dataset.get_task_num() + 1, 0);

  for (int i = 1; i < common_num_round + 1; ++i) {
    // clear gradient.
    gradient.clear();
    vector<float> pred_result;
    vector<vector<float>> preds(dataset.get_task_num() + 1);
    vector<float> eval_pred_result;
    vector<vector<float>> eval_preds(eval_set.get_task_num() + 1);
    common_trees[i - 1]->predict(dataset, pred_result);
    common_trees[i - 1]->predict(eval_set, eval_pred_result);
    // combine the predict result.
    // train set
    for (int j = 0; j < pred.size(); ++j) {
      pred[j] += pred_result[j];
      preds[task[j]].push_back(pred[j]);
    }
    // eval set
    for (int j = 0; j < eval_pred.size(); ++j) {
      eval_pred[j] += eval_pred_result[j];
      eval_preds[eval_task[j]].push_back(eval_pred[j]);
    }

    // 判断本次迭代是否满足early stop条件
    for (int j = 1; j <= dataset.get_task_num(); ++j) {
      if (flag[j]) {
        continue;
      }
      float loss_score = 0.0f;
      this->calculate_loss_score(eval_labels[j], eval_preds[j], eval_metric, j, loss_score);
      if (i == 1) {
        commmon_best_iterations.push_back(make_pair(i - 1, loss_score));
        common_min_loss_scores[j] = loss_score;
      } else {
        if (loss_score > common_min_loss_scores[j]) {
          common_accum_rounds[j] += 1;
          if (common_accum_rounds[j] == early_stopping_rounds) {
            flag[j] = true;
          }
        } else {
          common_accum_rounds[j] = 0;
          commmon_best_iterations[j - 1].first = i;
          commmon_best_iterations[j - 1].second = loss_score;
          common_min_loss_scores[j] = loss_score;
          // remove best_preds
          best_preds[j].resize(0);
          best_eval_preds[j].resize(0);
          for (int k = 0; k < preds[j].size(); ++k) {
            best_preds[j].push_back(preds[j][k]);
          }
          for (int k = 0; k < eval_preds[j].size(); ++k) {
            best_eval_preds[j].push_back(eval_preds[j][k]);
          }
        }
      }
    }

    // calculate new gradient.
    loss->get_gradient(pred, dataset.get_label_data(), gradient);
//    cout << "this is the gradient size: " << gradient.size() << endl;
    dataset.set_gradients(gradient);
    if (i == common_num_round) break;
    Tree *new_tree =
        new Tree(max_depth,
                 lambda,
                 beta,
                 dataset.get_feature_size(),
                 min_sample_leaf,
                 learning_rate,
                 regularization,
                 loss,
                 objective);
    cout << endl << "this is the " << i + 1 << "th round common train" << endl;
    new_tree->train(dataset);
    common_trees.push_back(new_tree);
  }


  if (max_num_round == common_num_round) {
    return 0;
  }

  // split dataset by task
  SingleTaskUpdater *single_objective = new SingleTaskUpdater;
  vector<Dataset> datasets;
  vector<Dataset> eval_datasets;
  for (int i = 0; i < dataset.get_task_num(); ++i) {
    Dataset tmp_dataset(dataset.get_feature_size());
    datasets.push_back(tmp_dataset);
    Dataset tmp_eval_dataset(eval_set.get_feature_size());
    eval_datasets.push_back(tmp_eval_dataset);
  }
  dataset.get_data_by_tasks(datasets);
  eval_set.get_data_by_tasks(eval_datasets);

  if (early_stopping_rounds != 0) {
    // update single_num_rounds for each task
    if (this->common_num_round != 0) {
      for (int i = 0; i < dataset.get_task_num(); ++i) {
        this->single_num_rounds[i] = max_num_round - commmon_best_iterations[i].first - 1;
      }
    }
    // update gradient to gradient of best iteration
    for (int i = 0; i < dataset.get_task_num(); ++i) {
      gradient.clear();
      loss->get_gradient(best_preds[i + 1], datasets[i].get_label_data(), gradient);
      datasets[i].set_gradients(gradient);
    };
  } else {
    for (int j = 0; j < dataset.get_task_num(); ++j) {
      best_preds[j + 1].resize(0);
      best_eval_preds[j + 1].resize(0);
    }
    for (int i = 0; i < dataset.get_data_size(); ++i) {
      best_preds[task[i]].push_back(pred[i]);
    }
    for (int i = 0; i < eval_set.get_data_size(); ++i) {
      best_eval_preds[eval_task[i]].push_back(eval_pred[i]);
    }
  }

  // init flag
  for (int i = 1; i <= flag.size(); ++i) {
    flag[i] = false;
  }

  // single training
  vector<float> single_pre_loss_scores(eval_set.get_task_num(), 1.0f);
  vector<vector<float> > preds(best_preds);
  vector<vector<float> > eval_preds(best_eval_preds);

  for (int i = 0; i < dataset.get_task_num(); ++i) {
    cout << endl << "this is the " << i + 1 << "th task single train" << endl;
    int accum_rounds = 0;
    for (int j = 0; j < this->single_num_rounds[i] + 1; ++j) {
      if (j == 0) {
        vector<Tree *> single_tree;
        single_trees.push_back(single_tree);
      } else {
        gradient.clear();
        vector<float> pred_result;
        vector<float> eval_pred_result;
        single_trees[i][j - 1]->predict(datasets[i], pred_result);
        single_trees[i][j - 1]->predict(eval_datasets[i], eval_pred_result);
        // combine the predict result.
        for (int k = 0; k < preds[i + 1].size(); ++k) {
          preds[i + 1][k] += pred_result[k];
        }
        for (int k = 0; k < eval_preds[i + 1].size(); ++k) {
          eval_preds[i + 1][k] += eval_pred_result[k];
        }
        float loss_score = 0.0f;
        this->calculate_loss_score(eval_datasets[i].get_label_data(),
                                   eval_preds[i + 1],
                                   eval_metric,
                                   i + 1,
                                   loss_score);
        if (j == 1) {
          single_best_iterations.push_back(make_pair(j - 1, loss_score));
          single_pre_loss_scores[i] = loss_score;
        } else {
          if (!flag[i + 1]) {
            if (loss_score >= single_pre_loss_scores[i]) {
              accum_rounds += 1;
              if (accum_rounds == early_stopping_rounds) {
                flag[i + 1] = true;
              }
            } else {
              accum_rounds = 0;
              single_best_iterations[i].first = j;
              single_best_iterations[i].second = loss_score;
              single_pre_loss_scores[i] = loss_score;
            }
          }
        }
        // calculate new gradient.
        loss->get_gradient(preds[i + 1], datasets[i].get_label_data(), gradient);
        datasets[i].set_gradients(gradient);
      }
      if (j == this->single_num_rounds[i]) break;
      Tree *new_tree =
          new Tree(max_depth,
                   lambda,
                   beta,
                   dataset.get_feature_size(),
                   min_sample_leaf,
                   learning_rate,
                   regularization,
                   loss,
                   single_objective);
      new_tree->train(datasets[i]);
      single_trees[i].push_back(new_tree);
      if (j != this->single_num_rounds[i]) {
        cout << "this is the " << j + 1 << "th round single train" << endl;
      }
    }
    cout << endl;
  }
  for (int i = 0; i < dataset.get_task_num(); ++i) {
    if (this->common_num_round != 0) {
      cout << "common train stage: the best iteration of the " << i + 1 << "th task is "
           << commmon_best_iterations[i].first << endl;
    }
    if (this->single_num_rounds[i] != 0) {
      cout << "single train stage: the best iteration of the " << i + 1 << "th task is "
           << single_best_iterations[i].first << endl;
    }
  }
  return 0;
}
template<typename LOSS, typename UPDATER>
int Booster<LOSS, UPDATER>::predict(Dataset &dataset, vector<float> &score, const string &log_path) {
  vector<Dataset> datasets;
  for (int i = 0; i < dataset.get_task_num(); ++i) {
    Dataset tmp_dataset(dataset.get_feature_size());
    datasets.push_back(tmp_dataset);
  }
  dataset.get_data_by_tasks(datasets);
//  time_t now = time(NULL);
//  ofstream ofile(log_path + "log_score_var", ios::app);
  for (int i = 0; i < dataset.get_task_num(); ++i) {
    vector<float> preds(datasets[i].get_data_size(), 0.0);
    float loss_score = 0.0f;
    single_predict(datasets[i], preds, i, loss_score);
    cout << endl;
    if (this->eval_metric == "auc") {
      loss_score = 1 - loss_score;
    }
    score.push_back(loss_score);
//    ofile << asctime(localtime(&now)) << "The " << i + 1 << "th task " <<this->eval_metric<< "score is " << loss_score << endl;
  }
//  ofile<<endl;
//  ofile.close();
  return 0;
}

template<typename LOSS, typename UPDATER>
int Booster<LOSS, UPDATER>::single_predict(const Dataset &dataset,
                                           vector<float> &pred,
                                           const int &task_id,
                                           float &loss_score) {
  if (this->common_num_round != 0) {
    int iteration =
        this->early_stopping_rounds != 0 ? this->commmon_best_iterations[task_id].first : this->common_num_round;
    for (int i = 0; i < iteration; ++i) {
      cout << "common stage: this is the " << i + 1 << "th round predict" << endl;
      vector<float> pred_result;
      common_trees[i]->predict(dataset, pred_result);
      // combine the predict result.
      for (int j = 0; j < pred.size(); ++j) {
        pred[j] += pred_result[j];
      }
      this->calculate_loss_score(dataset.get_label_data(), pred, eval_metric, task_id + 1, loss_score);
    }
  }
  if (this->single_num_rounds[task_id] != 0) {
    int iteration =
        this->early_stopping_rounds != 0 ? this->single_best_iterations[task_id].first
                                         : this->single_num_rounds[task_id];
    for (int i = 0; i < iteration; ++i) {
      cout << "single stage: this is the " << i + 1 << "th round predict" << endl;
      vector<float> pred_result;
      single_trees[task_id][i]->predict(dataset, pred_result);
      for (int j = 0; j < pred.size(); ++j) {
        pred[j] += pred_result[j];
      }
      this->calculate_loss_score(dataset.get_label_data(), pred, eval_metric, task_id + 1, loss_score);
    }
  }
  return 0;
}

template<typename LOSS, typename UPDATER>
int Booster<LOSS, UPDATER>::calculate_loss_score(const vector<float> &label,
                                                 const vector<float> &pred,
                                                 const string &eval_metric,
                                                 const int &task_id,
                                                 float &loss_score) {
  if (eval_metric == "logloss") {
    loss_score = BinaryLogLoss(label, pred);
    cout << "this is the " << task_id << "th task logloss: " << loss_score << endl;
  } else if (eval_metric == "auc") {
    loss_score = 1 - AUC(label, pred);
    cout << "this is the " << task_id << "th task auc: " << 1 - loss_score << endl;
  } else if (eval_metric == "rmse") {
    loss_score = RMSE(label, pred);
    cout << "this is the " << task_id << "th task rmse: " << loss_score << endl;
  } else if (eval_metric == "nrmse") {
    loss_score = nrMSE(label, pred);
    cout << "this is the " << task_id << "th task nrmse: " << loss_score << endl;
  }
}
