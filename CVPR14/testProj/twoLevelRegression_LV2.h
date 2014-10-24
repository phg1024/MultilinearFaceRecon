#pragma once

#include "TwoLevelRegression.h"

namespace ESRAligner {

  class TwoLevelRegression_LV2 : public TwoLevelRegression
  {
  public:
    TwoLevelRegression model_lv2;
    void loadFull(char *modelLV1, char *modelLV2);
    bool predict_real_lv2(IplImage *img, int sampleNum);
  };

}