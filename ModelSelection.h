//
// Created by zebang.zhzb on 2018/7/14.
//

#ifndef MTREE_MODELSELECTION_H
#define MTREE_MODELSELECTION_H

#include "Dataset.h"
#include "utils.h"
#include "Booster.h"
#include "LinearLoss.h"
#include "MultiTaskUpdater.h"

template<typename PARAM_TYPE>
class ModelSelection {
 public:
  /*! \belief: select the best model. */
  Booster<LogisticLoss, MultiTaskUpdater> get_best_booster(Dataset &dataset,
                                                           const vector<PARAM_TYPE> &params,
                                                           const string &param_name,
                                                           const int &max_num_round,
                                                           const int &common_num_round,
                                                           const float &beta) const;

};

#endif //MTREE_MODELSELECTION_H
