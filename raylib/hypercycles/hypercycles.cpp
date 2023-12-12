#include "hypercycles.h"
#include <string>

void Hypercycles::setup(int _screenWidth, int _screenHeight) {
    screenWidth = _screenWidth;
    screenHeight = _screenHeight;

    size_t sizeOfData = screenWidth * screenHeight * sizeof(bool);
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_grid_in, sizeOfData);
    cudaMallocManaged(&d_grid_out, sizeOfData);

    size_t sizeOfColorData = screenWidth * screenHeight * sizeof(RGBA);
    cudaMallocManaged(&h_grid_out, sizeOfColorData);

    // initialize grid on the host
    for (int i = 0; i < screenWidth * screenHeight; i++) {
        // initialise initial grid randomly
        d_grid_in[i] = GetRandomValue(0, 1) == 1;
        d_grid_out[i] = false;
    }

    // initialize opengl
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        // Problem: glewInit failed, something is seriously wrong.
        printf("Error: %s\n", glewGetErrorString(err));
    }

    // initialize opengl buffer
    glGenBuffers(1, &gl_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, gl_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeOfColorData, nullptr, GL_DYNAMIC_DRAW);

    // register buffer with cuda
    cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_buffer, cudaGraphicsMapFlagsWriteDiscard);

    // create texture
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters as needed
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Allocate texture storage (without specifying data)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenWidth, screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);


    texture = { 0 };
    texture.id = textureID; // OpenGL texture ID
    texture.width = screenWidth;
    texture.height = screenHeight;
    texture.mipmaps = 1;
    texture.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8; // Match this with your data format
}

void Hypercycles::update() {
    std::string updateMethod = "cuda";
    if (updateMethod == "cpp") {
        updateCpp();
    }
    else if (updateMethod == "cuda") {
        updateCuda();
    }
}

void Hypercycles::updateCpp() {

}

void Hypercycles::updateCuda() {
    Automata::updateCA(d_grid_in, d_grid_out, screenWidth, screenHeight);
    Automata::updateCAColour(d_grid_out, h_grid_out, screenWidth, screenHeight);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Map buffer object for writing from CUDA
    cudaGraphicsMapResources(1, &cuda_resource, 0);
    size_t num_bytes; 
    void *d_ptr;
    cudaGraphicsResourceGetMappedPointer(&d_ptr, &num_bytes, cuda_resource);

    // copy the data from cuda to opengl buffer
    // Copy data from CUDA buffer to OpenGL buffer
    size_t sizeOfColorData = screenWidth * screenHeight * sizeof(RGBA);
    cudaMemcpy(d_ptr, h_grid_out, sizeOfColorData, cudaMemcpyDeviceToDevice);

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);

    // update texture
    // Bind the buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_buffer);

    // Map the entire buffer
    // void* mappedBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

    // // Example for a buffer holding float data
    // bool* boolBuffer = static_cast<bool*>(mappedBuffer);

    // glUnmapBuffer(GL_ARRAY_BUFFER);

    // Update the texture with the buffer data
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screenWidth, screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Unbind the buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void Hypercycles::draw() {
    // Update
    //----------------------------------------------------------------------------------
    BeginDrawing();

    ClearBackground(RAYWHITE);

    // Draw the texture
    DrawTexture(texture, 0, 0, WHITE);

    EndDrawing();
}

void Hypercycles::cleanup() {
    cudaFree(d_grid_in);
    cudaFree(d_grid_out);

    glDeleteTextures(1, &texture.id);

}