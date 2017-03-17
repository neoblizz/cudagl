
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "render.h"
#include "tools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define BACKFACECULLING 1
#define DEBUG_STATISTICS 0 

glm::vec3* framebuffer;
fragment* depthbuffer;
unsigned int* depth;
float* device_vbo;
float* device_cbo;
int* device_ibo;
int* numCulledTriangles;
float* device_vbo_eye;
float* device_nbo;
vertex* vertices;
triangle* primitives;

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

__host__ __device__ glm::vec3 reflect(glm::vec3 I, glm::vec3 N)
{
	glm::vec3 R = I - 2.0f * N * glm::dot(I, N);
	return R;
}

__host__ __device__ void printVec4(glm::vec4 m){

	printf("%f, %f, %f, %f;\n", m[0], m[1], m[2], m[3]);
}

__host__ __device__ void printVec3(glm::vec3 m){

	printf("%f, %f, %f;\n", m[0], m[1], m[2]);
}


__host__ __device__ void printMat4(glm::mat4 m){
    printf("%f, %f, %f, %f;\n", m[0][0], m[1][0], m[2][0], m[3][0]);
    printf("%f, %f, %f, %f;\n", m[0][1], m[1][1], m[2][1], m[3][1]);
    printf("%f, %f, %f, %f;\n", m[0][2], m[1][2], m[2][2], m[3][2]);
    printf("%f, %f, %f, %f;\n", m[0][3], m[1][3], m[2][3], m[3][3]);
}

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    depthbuffer[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, unsigned int* depth, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
      depth[index] = 0;
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }

      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}


__global__ void vertexShadeKernel(glm::mat4 view, glm::mat4 projection, glm::vec3 light, float* vbo, int vbosize, float* nbo, int nbosize, vertex* vertices ){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  glm::vec4 point;
  glm::vec4 point_tformd;
  glm::vec3 normal;
  if(index<vbosize/3){
    // Assemble vec4 from vbo ... vertex assembly :)
    point.x = vbo[3*index];
    point.y = vbo[3*index+1];
    point.z = vbo[3*index+2];
    point.w = 1.0f;

    normal.x = nbo[3*index];
    normal.y = nbo[3*index+1];
    normal.z = nbo[3*index+2];

    // Model identity matrix
    glm::mat4 model = glm::mat4( 1.0f );

    // Before transforming compute light direction for each vertex
    vertices[index].lightdir = glm::normalize(light - glm::vec3( point.x, point.y, point.z ));
    // Copy over normals
    vertices[index].normal = normal;

    // Transform vertex and normal
    point_tformd = projection*view*model*point;

    // Convert to normalized device coordinates
    float div = 1.0f;
    if ( abs(point_tformd.w) > 0.0f )
      div = 1.0f/point_tformd.w;

    point.x = point_tformd.x*div;
    point.y=  point_tformd.y*div;
    point.z = point_tformd.z*div;

    // Do clipping on vertex
    vertices[index].draw_flag = true;
    if ( abs(point.x) >= 1.0 || abs(point.y) >= 1.0 || abs(point.z) >= 1.0 )
      vertices[index].draw_flag = false;
    vertices[index].point.x = point.x;
    vertices[index].point.y = point.y;
    vertices[index].point.z = point.z;
  }
}


__global__ void primitiveAssemblyKernel(vertex* vertices, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  int i0;
  int i1;
  int i2;
  if(index<primitivesCount){
    // Pull out indices
    i0 = ibo[3*index];
    i1 = ibo[3*index+1];
    i2 = ibo[3*index+2];

    // If any vertices should be drawn then draw the whole triangle
    primitives[index].toBeDiscard = true;
    if ( (vertices[i0].draw_flag + vertices[i1].draw_flag + vertices[i2].draw_flag) != 0 )
      primitives[index].toBeDiscard = false;

    // Copy over vertex points
    primitives[index].p0 = vertices[i0].point;
    primitives[index].p1 = vertices[i1].point;
    primitives[index].p2 = vertices[i2].point;

    // Copy over normals
    primitives[index].eyeNormal0 = vertices[i0].normal;
    primitives[index].eyeNormal1 = vertices[i1].normal;
    primitives[index].eyeNormal2 = vertices[i2].normal;

    // Copy over vertex colors
    primitives[index].c0 = glm::vec3( cbo[0], cbo[1], cbo[2] );
    primitives[index].c1 = glm::vec3( cbo[3], cbo[4], cbo[5] );
    primitives[index].c2 = glm::vec3( cbo[6], cbo[7], cbo[8] );

    // Copy over light directions
    primitives[index].eyeCoords0 = vertices[i0].lightdir;
    primitives[index].eyeCoords1 = vertices[i1].lightdir;
    primitives[index].eyeCoords2 = vertices[i2].lightdir;
  }
}

__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, unsigned int* depth, glm::vec2 resolution, int* numCulledTriangles) {

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  glm::vec2 p0;
  glm::vec2 p1;
  glm::vec2 p2;
  glm::vec3 min_point;
  glm::vec3 max_point;

  triangle tri;
  glm::vec3 bary_coord;

  fragment frag;

  float scale_x;
  float scale_y;
  float offs_x;
  float offs_y;
  if ( index<primitivesCount ) {

    // Check if we should even bother with this triangle
    if ( primitives[index].toBeDiscard ) {
#if DEBUG_STATISTICS
	if (index == 0) *numCulledTriangles = 0;
	__syncthreads();
	atomicAdd(numCulledTriangles, 1);
#endif
	return;
    }

    // Map primitives from world to window coordinates using the viewport transform
    scale_x = resolution.x/2;
    scale_y = resolution.y/2;
    offs_x = resolution.x/2;
    offs_y = resolution.y/2;

    tri.p0.x = scale_x*primitives[index].p0.x + offs_x;
    tri.p1.x = scale_x*primitives[index].p1.x + offs_x;
    tri.p2.x = scale_x*primitives[index].p2.x + offs_x;

    tri.p0.y = offs_y - scale_y*primitives[index].p0.y;
    tri.p1.y = offs_y - scale_y*primitives[index].p1.y;
    tri.p2.y = offs_y - scale_y*primitives[index].p2.y;

    // Backface culling
#if BACKFACECULLING
    if ( calculateSignedArea( primitives[index] ) > 0.0f )
      return;
#endif

    // Bounding box
    getAABBForTriangle( tri, min_point, max_point );

    // Ensure window bounds are maintained
    min_point.x = max( min_point.x, 0.0f );
    min_point.y = max( min_point.y, 0.0f );
    max_point.x = min( max_point.x, resolution.x );
    max_point.y = min( max_point.y, resolution.y );

    // For each pixel in the bounding box check if its in the triangle
    for ( int x=glm::floor(min_point.x); x<glm::ceil(max_point.x); ++x ) {
      for ( int y=glm::floor(min_point.y); y<glm::ceil(max_point.y); ++y ) {
	int frag_index = x + (y * resolution.x);
	bary_coord = calculateBarycentricCoordinate( tri, glm::vec2( x,y ) );
	if ( isBarycentricCoordInBounds( bary_coord ) ) {

	  // Color a fragment just for debugging sake
	  frag.position = getXYZAtCoordinate( bary_coord, primitives[index] );
	  frag.normal = bary_coord[0]*primitives[index].eyeNormal0 \
		      + bary_coord[1]*primitives[index].eyeNormal1 \
		      + bary_coord[2]*primitives[index].eyeNormal2;

	  // Correct color interpolation on triangle
	  frag.color = bary_coord[0]*primitives[index].c0 \
		     + bary_coord[1]*primitives[index].c1 \
		     + bary_coord[2]*primitives[index].c2;

	  frag.lightdir = bary_coord[0]*primitives[index].eyeCoords0 \
		        + bary_coord[1]*primitives[index].eyeCoords1 \
		        + bary_coord[2]*primitives[index].eyeCoords2;

	  // Block until its our turn to do a compare
	  while ( !atomicCAS( &depth[frag_index], 0, 1 ) );
	  fragment cur_frag = getFromDepthbuffer( x, y, depthbuffer, resolution );
	  // If current value is gt than new value then update
	  if ( frag.position.z > cur_frag.position.z )
	    writeToDepthbuffer( x, y, frag, depthbuffer, resolution );
	  // Release lock
	  depth[frag_index] = 0;
	}
      }
    }
  }
}

//Implement a fragment shader
//specular lighting, diffuse lighting, ambient lighting
//source opengl-tutoral.org
__global__ void fragmentShadeKernel( fragment* depthbuffer, glm::vec2 resolution, int draw_mode ){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  // int index = x + (y * resolution.x);

  fragment frag;

  if(x<=resolution.x && y<=resolution.y){
    frag = getFromDepthbuffer( x, y, depthbuffer, resolution );

    // Interactive drawing modes
    switch (draw_mode) {
      case( DRAW_SOLID ):
	if ( glm::length( frag.color ) > 1e-10 )
	  frag.color = glm::vec3( 1.0, 1.0, 1.0 );
	break;
      case( DRAW_COLOR ):
	// Keep color the same
	break;
      case( DRAW_NORMAL ):
	frag.color = frag.normal;
	break;
      case( SHADE_SOLID ):
	// Lambertian shading
	if ( glm::length( frag.color ) > 1e-10 )
	  frag.color = glm::vec3( 1.0, 1.0, 1.0 );
	frag.color = clamp(max(glm::dot( frag.normal, frag.lightdir ), 0.0f), 0.0f, 1.0f)*frag.color;
	break;
      case( SHADE_COLOR ):
	// Lambertian shading
	frag.color = clamp(max(glm::dot( frag.normal, frag.lightdir ), 0.0f), 0.0f, 1.0f)*frag.color;
	break;
    }
    writeToDepthbuffer( x, y, frag, depthbuffer, resolution );
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(glm::mat4 view, glm::mat4 projection, glm::vec3 light, int draw_mode, uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));

  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  depth = NULL;
  cudaMalloc((void**)&depth, (int)resolution.x*(int)resolution.y*sizeof(unsigned int));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));

  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.lock = 0;
  frag.z = -FLT_MAX;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, depth, frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  vertices = NULL;
  cudaMalloc((void**)&vertices, (ibosize)*sizeof(vertex));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_vbo_eye = NULL;
  cudaMalloc((void**)&device_vbo_eye, vbosize*sizeof(float));

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

#if DEBUG_STATISTICS
  numCulledTriangles = NULL;
  cudaMalloc((void**)&numCulledTriangles, sizeof(int));
#endif

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(view, projection, light, device_vbo, vbosize, device_nbo, nbosize, vertices);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(vertices, device_cbo, cbosize, device_ibo, ibosize, primitives);
  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, depth, resolution, numCulledTriangles);
#if DEBUG_STATISTICS
  int * host_CulledT;
  cudaHostAlloc((void**) &host_CulledT, sizeof(int), cudaHostAllocDefault);
  cudaMemcpy( host_CulledT, numCulledTriangles, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Number of Culled Triangles: %d", *host_CulledT);
#endif
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, draw_mode);
  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( vertices );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( device_nbo );
  cudaFree( device_vbo_eye );
  cudaFree( depth );
}
