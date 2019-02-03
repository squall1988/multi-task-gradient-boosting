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
  clock_t start = clock();

  int n_splits = 10;
  Booster<LinearLoss, MultiTaskUpdater>
      booster(max_num_round, common_num_round, 5, 0.1, beta, 10, learning_rate, regularization);
//  vector<pair<Dataset, Dataset>> datasets = data.shuffle_split(n_splits, 0.25, 33);
  Matrix scores;
  for (int i = 0; i < n_splits; ++i) {
    Dataset train_data = load_dataset(path + "train_re" + to_string(i+1) + ".csv", feature_size, task_num);
    Dataset test_data = load_dataset(path + "test_re" + to_string(i+1) + ".csv", feature_size, task_num);
    booster.train(train_data, train_data, eval_metric, early_stopping_rounds, false);
    vector<float> score;
    booster.predict(test_data, score, log_path);
    scores.push_back(score);;
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
  clock_t finish = clock();
  cout << "program cost time: " << (double)(finish - start)/CLOCKS_PER_SEC <<"seconds" << endl;
  ofile << "program cost time: " << (double)(finish - start)/CLOCKS_PER_SEC <<"seconds" << endl;
  ofile << "------------------------------------------------" << endl;
  ofile.close();
  return SUCCESS;
}

int single_sarcos_boost() {
  string eval_metric = "nrmse";
  string dataset_name = "sarcos";
  int feature_size = 21;
  int task_num = 7;
  string log_path = "D:\\C++\\ClionProject\\multi-task-gradient-boosting\\data\\sarcos\\";
  string path = "D:\\C++\\ClionProject\\multi-task-gradient-boosting\\data\\sarcos\\sub_sarcos.csv";
  int max_num_round = 500;
  int common_num_round = 500;
  float beta = 0;
  int early_stopping_round = 10;
  float learning_rate = 0.05;
  string regularization = "entropy";

  boost(max_num_round,
        common_num_round,
        beta,
        early_stopping_round,
        eval_metric,
        dataset_name,
        log_path,
        path,
        feature_size,
        task_num,
        learning_rate,
        regularization
  );
  return SUCCESS;
}



int single_school_boost() {
  string eval_metric = "rmse";
  string dataset_name = "school";
  int feature_size = 28;
  int task_num = 139;
  string log_path = "D:\\C++\\ClionProject\\multi-task-gradient-boosting\\data\\school\\";
  string path = "D:\\C++\\ClionProject\\multi-task-gradient-boosting\\data\\school\\school_";
  int max_num_round = 500;
  int common_num_round = 500;
  float beta = 0.01;
  int early_stopping_round = 10;
  float learning_rate = 0.05;
  string regularization = "variance";
  boost(max_num_round,
        common_num_round,
        beta,
        early_stopping_round,
        eval_metric,
        dataset_name,
        log_path,
        path,
        feature_size,
        task_num,
        learning_rate,
        regularization
  );
  return SUCCESS;
}


int test_class_boost() {
  vector<int> common_num_rounds{0};
  vector<float> betas{0, 0.001, 1.0};
  vector<int> early_stopping_rounds{0};
  float learning_rate = 0.1;
  Booster<LogisticLoss, MultiTaskUpdater>
      booster(20, 10, 5, 0.1, betas[0], 10, learning_rate, "variance");
  Dataset data = load_dataset("/Users/squall/work/tree/data/xijue_data.txt", 263, 4);
  booster.train(data, data, "auc", 4, false);
  return SUCCESS;

}

int main() {
  single_school_boost();
  return 0;
}