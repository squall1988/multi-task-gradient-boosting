//
// Created by zebang.zhzb on 2018/7/14.
//

#include "ModelSelection.h"

template<typename PARAM_TYPE>
Booster<LogisticLoss, MultiTaskUpdater> ModelSelection<PARAM_TYPE>::get_best_booster(Dataset &dataset,
                                                                                     const vector<PARAM_TYPE> &params,
                                                                                     const string &param_name,
                                                                                     const int &max_num_round,
                                                                                     const int &common_num_round,
                                                                                     const float &beta) const {
  for (int i = 0; i < params.size(); ++i) {
    if (param_name == "common_num_round") {
      Booster<LogisticLoss, MultiTaskUpdater> booster(max_num_round, params[i], 5, 0.1, beta, 10, 0.05);
      pair<Dataset, Dataset> p = dataset.train_test_split(0.3, 33);
      booster.train(p.first, p.second, "auc", 0, false);
      vector<float> score;
      booster.predict(p.second, score, NULL);
    } else if (param_name == "beta") {
      Booster<LogisticLoss, MultiTaskUpdater> booster(max_num_round, common_num_round, 5, 0.1, params[i], 10, 0.05);
      pair<Dataset, Dataset> p = dataset.train_test_split(0.3, 33);
      booster.train(p.first, p.second, "auc", 0, false);
      vector<float> score;
      booster.predict(p.second, score, NULL);
    }
  }
};