#pragma once

#include <string>

#include <GL/glew.h>
#include <GL/gl.h>
#include "cuda_gl_interop.h"
#include "raylib.h"

#include "automata.cuh"
#include "structs.h"

class Hypercycles
{
    public:
        void setup(int screenWidth, int screenHeight);
        void update();
        void draw();
        void cleanup();

        void updateCpp();
        void updateCuda();

        bool* d_grid_in;
        bool* d_grid_out;
        RGBA* h_grid_out;
        Texture2D texture;
        GLuint gl_buffer;
        size_t size_of_data;
        cudaGraphicsResource_t cuda_resource;

        int screenWidth;
        int screenHeight;
};