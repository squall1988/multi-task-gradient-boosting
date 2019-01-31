#include <iostream>
#include <fstream>
#include <set>
#include <queue>
#include <ctime>
#include <iomanip>
#include "utils.h"
#include "tree.h"
#include "LinearLoss.h"
#include "SingleTaskUpdater.h"
#include "MultiTaskUpdater.h"
#include "Booster.h"
#include "Booster.cpp"
#include "Random.h"
#include "ThreadPool.h"

using namespace std;


Dataset load_dataset(string path, const int &feature_size, const int &task_num) {
  Dataset dataset(feature_size);
  dataset.set_task_num(task_num);
  cout << "begin load train data" << endl;
  ERROR_CODE_CHECK(dataset.load_data_from_file(
      path,
      ","));
  cout << "load train data successful" << endl;
  return dataset;
}

// single experiment
int boost(int max_num_round,
          int common_num_round,
          float beta,
          int early_stopping_rounds,
          string eval_metric,
          string dataset_name,
          string log_path,
          string path,
          int feature_size,
          int task_num,
          float learning_rate,
          string regularization) {
  Booster<LinearLoss, MultiTaskUpdater>
      booster(max_num_round, common_num_round, 5, 0.1, beta, 10, learning_rate, regularization);
  Dataset data = load_dataset(path, feature_size, task_num);
  int n_splits = 5;
//  vector<pair<Dataset, Dataset>> datasets = data.shuffle_split_by_size(n_splits, 50, 5000, 377);
  vector<pair<Dataset, Dataset>> datasets = data.shuffle_split(n_splits, 0.2, 377);
  Matrix scores;
  for (int i = 0; i < n_splits; ++i) {
    booster.train(datasets[i].first, datasets[i].second, eval_metric, early_stopping_rounds, false);
    vector<float> score;
    booster.predict(datasets[i].second, score, log_path);
    scores.push_back(score);
  }
  //calculate each task mean score
  Matrix scores_tran = transpose(scores);
  ofstream ofile(log_path + "log_score", ios::app);
  ofile << "dataset_name: " << dataset_name << endl;
  ofile << "Parameter:" << endl;
  ofile << "max_num_round = " << max_num_round << ", common_num_round = " << common_num_round << ", regularization = "
        << regularization;
  if (regularization == "variance" or regularization == "weight_entropy") {
    ofile << ", beta = " << beta;
  }
  ofile << ", early_stopping_rounds = " << early_stopping_rounds << ", eval_metric = " << eval_metric << ", n_splits = "
        << n_splits << endl;
  // the mean score of all tasks
  float avg_score = 0;
  for (int i = 0; i < scores_tran.size(); ++i) {
    float sum = std::accumulate(std::begin(scores_tran[i]), std::end(scores_tran[i]), 0.0f);
    float mean = sum / scores_tran[i].size();
    avg_score += mean;
    float stddev = 0.0f;
    for (vector<float>::const_iterator it = scores_tran[i].begin(); it != scores_tran[i].end(); ++it) {
      stddev += (*it - mean) * (*it - mean);
    }
    stddev = std::sqrt(stddev / (scores_tran[i].size() - 1));
    time_t now = time(NULL);
    cout << asctime(localtime(&now)) << "The " << i + 1 << "th task " << eval_metric << " score:  " << mean << ", +/- "
         << stddev << endl;
//    ofile << asctime(localtime(&now)) << "The " << i + 1 << "th task " << eval_metric << " score:  " << mean
//          << ", +/- "
//          << stddev << endl;
    ofile << std::setprecision(4) << mean << "\t";
  }
  ofile << endl;
  time_t now = time(NULL);
  cout << asctime(localtime(&now)) << eval_metric << " score:  " << avg_score / scores_tran.size() << endl;
  ofile << asctime(localtime(&now)) << eval_metric << " score:  " << avg_score / scores_tran.size() << endl;
  ofile << "------------------------------------------------" << endl;
  ofile.close();
  return SUCCESS;
}

int test_boost() {
  vector<int> max_common_num_round{20, 30, 50, 70};
  vector<float> betas{0.1};
  string eval_metric = "nrmse";
  string dataset_name = "school";
//  string regularization = "entropy";
  int feature_size = 28;
  int task_num = 28;
  // log output dir
  //path for loading data
//  string log_path = "/gruntdata/public/zebang.zhzb/work/tools/odps_clt/";
//  string path = "/gruntdata/public/zebang.zhzb/work/tools/odps_clt/mt_data_40000.csv";

  string log_path = "D:\\DLS\\Multi-task Learning\\project\\tree-model\\example-data\\" + dataset_name + "\\";
  string path =
      "D:\\DLS\\Multi-task Learning\\project\\tree-model\\example-data\\" + dataset_name + "\\data.csv";

  boost(100,
        0,
        0,
        0,
        eval_metric,
        dataset_name,
        log_path,
        path,
        feature_size,
        task_num,
        0.05,
        "false");
  boost(100,
        100,
        0,
        0,
        eval_metric,
        dataset_name,
        log_path,
        path,
        feature_size,
        task_num,
        0.05,
        "false");

  for (int k = 0; k < max_common_num_round.size(); ++k) {
    boost(100,
          max_common_num_round[k],
          0,
          0,
          eval_metric,
          dataset_name,
          log_path,
          path,
          feature_size,
          task_num,
          0.05,
          "entropy"
    );
    boost(100,
          max_common_num_round[k],
          0.1,
          0,
          eval_metric,
          dataset_name,
          log_path,
          path,
          feature_size,
          task_num,
          0.05,
          "variance"
    );
  }
}

int single_test_boost() {
  string eval_metric = "nrmse";
  string dataset_name = "sarcos";
  int feature_size = 21;
  int task_num = 7;
  string log_path = "/Users/squall/work/tree/multi-task-gradient-boosting/sarcos_logs";
  string path = "/Users/squall/work/tree/multi-task-gradient-boosting/data/sarcos.txt";
  int max_num_round = 10;
  vector<int> common_num_rounds{5};
  vector<float> betas{0, 0.001, 1.0};
  vector<int> early_stopping_rounds{0};
  float learning_rate = 0.1;
  boost(max_num_round,
        common_num_rounds[0],
        betas[1],
        early_stopping_rounds[0],
        eval_metric,
        dataset_name,
        log_path,
        path,
        feature_size,
        task_num,
        learning_rate,
        "variance"
  );
}

int main() {
//  test_boost();
  single_test_boost();
  return 0;
}