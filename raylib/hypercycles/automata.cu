#include "automata.cuh"

namespace Automata {
	void updateCA(bool* d_grid_in, bool* d_grid_out, int width, int height) {
		// Compute the number of blocks and threads per block
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Call the kernel
		updateCAKernel<<<numBlocks, threadsPerBlock>>>(d_grid_in, d_grid_out, width, height);
	}

	__global__ 
	void updateCAKernel(bool* d_grid_in, bool* d_grid_out, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return; // Check bounds

		// Compute the index in the array
		int index = y * width + x;

		// Count live neighbors
		int live_neighbors = 0;
		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				if (dx == 0 && dy == 0) continue; // Skip the cell itself

				int nx = x + dx;
				int ny = y + dy;

				// Check for boundary conditions
				if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
					live_neighbors += d_grid_in[ny * width + nx] ? 1 : 0;
				}
			}
		}

		// Apply the Game of Life rules
		bool cell_is_alive = d_grid_in[index];
		if (cell_is_alive) {
			d_grid_out[index] = (live_neighbors == 2 || live_neighbors == 3);
		} else {
			d_grid_out[index] = (live_neighbors == 3);
		}
	}

	void updateCAColour(bool* d_grid_in, RGBA* d_grid_out, int width, int height) {
		// Compute the number of blocks and threads per block
		dim3 threadsPerBlock(16, 16);
		dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
			(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

		// Call the kernel
		updateCAColourKernel<<<numBlocks, threadsPerBlock>>>(d_grid_in, d_grid_out, width, height);
	}

	__global__
	void updateCAColourKernel(bool* d_grid_in, RGBA* d_grid_out, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return; // Check bounds

		// Compute the index in the array
		int index = y * width + x;

		// Apply the Game of Life rules
		bool cell_is_alive = d_grid_in[index];
		if (cell_is_alive) {
			d_grid_out[index] = { 0, 255, 0, 255 };
		} else {
			d_grid_out[index] = { 0, 0, 0, 255 };
		}
	}
}