#include "utils.h"

float dot(valarray<float> a, valarray<float> b)
{
	return (a * b).sum();
}

valarray<float> min(valarray<float> arr, float sca)
{
	valarray<float> ret(arr.size());
	for (int i = 0; i < arr.size(); i++)
	{
		ret[i] = min(arr[i], sca);
	}
	return ret;
}

valarray<float> max(valarray<float> arr, float sca)
{
	valarray<float> ret(arr.size());
	for (int i = 0; i < arr.size(); i++)
	{
		ret[i] = max(arr[i], sca);
	}
	return ret;
}

valarray<float> toOneHot(int idx, int len)
{
	valarray<float> ret(0.0, len);
	ret[idx] = 1.0;
	return ret;
}

float clamp(float val, float a, float b)
{
	return min(max(val, b), a);
}
