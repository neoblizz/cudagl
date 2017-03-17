#ifndef RENDER_H
#define RENDER_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
unsigned int* depth;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_vbo_eye;
float* device_nbo;
vertex* vertices;
triangle* primitives;

// Drawing modes
enum { DRAW_SOLID, DRAW_COLOR, DRAW_NORMAL, SHADE_SOLID, SHADE_COLOR };

void kernelCleanup();
void cudaRasterizeCore(glm::mat4 view, glm::mat4 projection, glm::vec3 light, int draw_mode, uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize);

#endif //RENDER_H
