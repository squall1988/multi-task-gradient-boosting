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

void test_split() {
  string a = "what the fuck";
  string delimiter = " ";
  vector<string> split_result = common::Split(a, *delimiter.c_str());
  for (int i = 0; i < split_result.size(); i++) {
    cout << split_result[i] << endl;
  }
}

void test_read() {
  ifstream in_file("/home/squall/plot_line.py");
  if (!in_file.is_open()) {
    cout << "this is wrong" << endl;
  }
  string line;
  while (in_file >> line) {
    cout << line << endl;
  }
}

void test_set() {
  set<int> unique_num;
  for (int i = 0; i < 10; i++) {
    unique_num.insert(i);
  }
  for (set<int>::iterator it = unique_num.begin(); it != unique_num.end(); ++it) {
    cout << *it << endl;
  }

}

void test_create_vec() {
  vector<float> tmp{1.0f, 2.0f};
  for (auto x : tmp) {
    cout << x << endl;
  }
}

void test_case() {
  test_split();
  test_read();
  test_set();
  test_create_vec();
  std::cout << "Hello, World!" << std::endl;
}

void test_queue() {
  queue<int> test_queue;
  for (int i = 0; i < 10; ++i) {
    test_queue.push(i);
  }
  while (!test_queue.empty()) {
    cout << test_queue.front() << endl;
    cout << "size is " << test_queue.size() << endl;
    test_queue.pop();
    cout << "size is " << test_queue.size() << endl;

  }
}

void test_sort() {
  vector<pair<int, int> > t;
  t.emplace_back(make_pair(1, 6));
  t.emplace_back(make_pair(3, 4));
  t.emplace_back(make_pair(2, 5));
  t.emplace_back(make_pair(4, 7));
  sort(t.begin(), t.end(), cmp);
  for (int i = 0; i < t.size(); ++i) {
    cout << t[i].first << " " << t[i].second << endl;
  }
}

void test_zero() {
  float reg = 0.0123f;
  float beta = 0.0;
  cout << beta * reg << endl;
}

void test_random() {
  common::Random seed(33);
  for (int i = 0; i < 5; ++i) {
    int x = seed.NextInt(1, 100);
    cout << x << endl;
    common::Random random(x);
    vector<int> test_index = random.Sample(1000, 300);
    vector<int> train_index;
    int j = 0;
    for (int k = 0; k < 1000; ++k) {
      if (test_index[j] != k) {
        train_index.push_back(k);
      } else {
        ++j;
      }
    }
//    cout << train_index.size() << " " << test_index.size() << " " << 1000 << endl;

  }
}

void test_shuffle_split() {
  string path = "/gruntdata/public/zebang.zhzb/work/tools/odps_clt/";
  string train_path = path + "mt_data.csv";
  Dataset dataset(68);
  dataset.set_task_num(2);
  cout << "begin load train data" << endl;
  dataset.load_data_from_file(
      train_path,
      ",");
  cout << "load train data successful" << endl;
  dataset.shuffle_split(3, 0.3, 33);
}

void test_mat_transpose() {
  Matrix mat(2, vector<float>(4, 0));
  int k = 1;
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < mat[i].size(); ++j) {
      mat[i][j] = k++;
      cout << mat[i][j] << " ";
    }
    cout << endl;
  }
  Matrix mat_tran = transpose(mat);
  for (int i = 0; i < mat_tran.size(); ++i) {
    for (int j = 0; j < mat_tran[i].size(); ++j) {
      cout << mat_tran[i][j] << " ";
    }
    cout << endl;
  }
}

int cp(int *x, int i, int k) {
  x[i] = k;
  return 0;
}

int mul(int x, int y) {
  return x * y;
}

//int test_threadpool() {
//  std::threadpool pool(4);
//  std::vector<std::future<float> > results;
//  auto *x = new int[100]();
//  for(int i=0;i<100;i++){
//    x[i] = -999;
//  }
//  int max_num = 0;
//  for (int i = 0; i < 100; ++i) {
//    results.emplace_back(
//        pool.commit(cp, x, i, i)
//    );
//  }
//  for(auto && result: results){
//    if(result.get() != 0){
//      return -1;
//    }
//  }
//  for(int i=0;i<100;++i){
//    max_num = x[i] > max_num ? x[i] : max_num;
//  }
//  cout<<"max_num: "<<max_num<<endl;
//  delete x;
//}


//int build_tree() {
//  SingleTaskUpdater objective;
//  LinearLoss *loss = new LinearLoss;
//  Tree tree_base(5, 0.1, 82, 20, 0.1, 0.1, loss, &objective);
//  Dataset dataset(82);
//  cout << "begin load data" << endl;
//  ERROR_CODE_CHECK(dataset.load_data_from_file("/home/squall/test_tree.csv", ","));
//  cout << "load data successful" << endl;
//  vector<float> pred(dataset.get_data_size(), 0.5);
//  Matrix gradient;
//  loss->get_gradient(pred, dataset.get_label_data(), gradient);
////    for (int i = 0; i < pred.size(); ++i) {
////        cout << pred[i] << endl;
////    }
//  dataset.set_gradients(gradient);
//  tree_base.train(dataset);
//  vector<float> pred_result;
//  tree_base.predict(dataset, pred_result);
////  for (auto x : pred_result) {
////    cout << x << endl;
////  }
//  return SUCCESS;
//}

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
          string regularization) {
  Booster<LinearLoss, MultiTaskUpdater>
      booster(max_num_round, common_num_round, 5, 0.1, beta, 10, 0.05, regularization);
  Dataset data = load_dataset(path, feature_size, task_num);
  int n_splits = 5;
  vector<pair<Dataset, Dataset>> datasets = data.shuffle_split(n_splits, 0.6, 377);
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
          "variance"
    );
  }
}

int single_test_boost() {
  string eval_metric = "nrmse";
  string dataset_name = "sarcos";
  int feature_size = 21;
  int task_num = 7;
  string log_path = "D:\\DLS\\Multi-task Learning\\project\\tree-model\\example-data\\" + dataset_name + "\\";
  string path =
      "D:\\DLS\\Multi-task Learning\\project\\tree-model\\example-data\\" + dataset_name + "\\data.csv";
  int max_num_round = 100;
  vector<int> common_num_rounds{100};
  vector<float> betas{0, 0.001, 1.0};
  vector<int> early_stopping_rounds{0};
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
        "variance"
  );
}

int main() {
//  test_boost();
  single_test_boost();
  return 0;
}