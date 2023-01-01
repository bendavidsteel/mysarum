#pragma once
#include "ofMain.h"

float dot(valarray<float> a, valarray<float> b);
valarray<float> min(valarray<float> arr, float sca);
valarray<float> max(valarray<float> arr, float sca);
valarray<float> toOneHot(int idx, int len);
float clamp(float val, float min, float max);