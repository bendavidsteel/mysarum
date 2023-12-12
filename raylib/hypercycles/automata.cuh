#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "structs.h"

namespace Automata {
	void updateCA(bool* d_grid_in, bool* d_grid_out, int width, int height);
	__global__ void updateCAKernel(bool* d_grid_in, bool* d_grid_out, int width, int height);
	void updateCAColour(bool* d_grid_in, RGBA* h_grid_out, int width, int height);
	__global__ void updateCAColourKernel(bool* d_grid_in, RGBA* d_grid_out, int width, int height);
}