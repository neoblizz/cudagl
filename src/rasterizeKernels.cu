
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define BACKFACECULLING 0

glm::vec3* framebuffer;
fragment* depthbuffer;
unsigned int* depth;
float* device_vbo;
float* device_cbo;
int* device_ibo;
float* device_vbo_eye;
float* device_nbo;
triangle* primitives;


void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }
}

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
      f.normal = glm::vec3(0.0f);
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


__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::vec2 resolution, glm::mat4 projection, glm::mat4 view, float zNear, float zFar, float *vbo_eye, float *nbo, int nbosize){

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
    //vertex assembly
    glm::vec4 vertex(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1.0f);
    glm::vec4 normal(nbo[3*index], nbo[3*index+1], nbo[3*index+2], 1.0f);
	
	printf("screen space coordinate:\n");
	printVec4(vertex);

	// transform position to eye space
	vertex = view*vertex;
	vbo_eye[3*index]   = vertex.x;
	vbo_eye[3*index+1] = vertex.y;
	vbo_eye[3*index+2] = vertex.z;

	// transform normal to eye space
	normal = glm::transpose(glm::inverse(view))*normal;
	nbo[3*index]   = normal.x;
	nbo[3*index+1] = normal.y;
	nbo[3*index+2] = normal.z;

	// project to clip space
	vertex = projection* vertex;
	// transform to NDC
	vertex /= vertex.w;
	// viewport transform
	vbo[3*index]   = resolution.x * 0.5f * (vertex.x + 1.0f);
	vbo[3*index+1] = resolution.y * 0.5f * (vertex.y + 1.0f);
	vbo[3*index+2] = (zFar-zNear)*0.5f*vertex.z + (zFar+zNear)*0.5f;

  }
}


__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float *vbo_eye, float *nbo, int nbosize, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	primitives[index].p0.x = vbo[3*ibo[3*index]];
	primitives[index].p0.y = vbo[3*ibo[3*index]+1];
	primitives[index].p0.z = vbo[3*ibo[3*index]+2];

	primitives[index].p1.x = vbo[3*ibo[3*index+1]];
	primitives[index].p1.y = vbo[3*ibo[3*index+1]+1];
	primitives[index].p1.z = vbo[3*ibo[3*index+1]+2];


	primitives[index].p2.x = vbo[3*ibo[3*index+2]];
	primitives[index].p2.y = vbo[3*ibo[3*index+2]+1];
	primitives[index].p2.z = vbo[3*ibo[3*index+2]+2];

	primitives[index].eyeCoords0.x = vbo_eye[3*ibo[3*index]];
	primitives[index].eyeCoords0.y = vbo_eye[3*ibo[3*index]+1];
	primitives[index].eyeCoords0.z = vbo_eye[3*ibo[3*index]+2];

	primitives[index].eyeCoords1.x = vbo_eye[3*ibo[3*index+1]];
	primitives[index].eyeCoords1.y = vbo_eye[3*ibo[3*index+1]+1];
	primitives[index].eyeCoords1.z = vbo_eye[3*ibo[3*index+1]+2];

	primitives[index].eyeCoords2.x = vbo_eye[3*ibo[3*index+2]];
	primitives[index].eyeCoords2.y = vbo_eye[3*ibo[3*index+2]+1];
	primitives[index].eyeCoords2.z = vbo_eye[3*ibo[3*index+2]+2];

	primitives[index].eyeNormal0.x = nbo[3*ibo[3*index]];
	primitives[index].eyeNormal0.y = nbo[3*ibo[3*index]+1];
	primitives[index].eyeNormal0.z = nbo[3*ibo[3*index]+2];

	primitives[index].eyeNormal1.x = nbo[3*ibo[3*index+1]];
	primitives[index].eyeNormal1.y = nbo[3*ibo[3*index+1]+1];
	primitives[index].eyeNormal1.z = nbo[3*ibo[3*index+1]+2];

	primitives[index].eyeNormal2.x = nbo[3*ibo[3*index+2]];
	primitives[index].eyeNormal2.y = nbo[3*ibo[3*index+2]+1];
	primitives[index].eyeNormal2.z = nbo[3*ibo[3*index+2]+2];

	primitives[index].c0.x = cbo[0];
	primitives[index].c0.y = cbo[1];
	primitives[index].c0.z = cbo[2];

	primitives[index].c1.x = cbo[3];
	primitives[index].c1.y = cbo[4];
	primitives[index].c1.z = cbo[5];

	primitives[index].c2.x = cbo[6];
	primitives[index].c2.y = cbo[7];
	primitives[index].c2.z = cbo[8];

	primitives[index].toBeDiscard = 0;

#if defined(BACKFACECULLING)
   if(calculateSignedArea(primitives[index]) < 1e-6) {
      primitives[index].toBeDiscard = 1; // back facing triangles
   }else{
      glm::vec3 triMin, triMax;
      getAABBForTriangle(primitives[index], triMin, triMax);
         if(triMin.x > resolution.x || triMin.y > resolution.y || triMin.z > zFar ||
           triMax.x < 0|| triMax.y < 0 || triMax.z < zNear)
           primitives[index].toBeDiscard = 1;
   }

#endif
  }
}

//DONE: Implement a rasterization method, such as scanline.
/*
   Given triangle coordinates, converted to screen coordinates, find fragments inside of triangle using AABB and brute force barycentric coords checks
*/
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, unsigned int* depth, glm::vec2 resolution) {

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
    if ( !primitives[index].toBeDiscard )
      return;

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
    if ( calculateSignedArea( primitives[index] ) > 0.0f )
      return;

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

	  /*frag.lightdir = bary_coord[0]*primitives[index].eyeCoords0 \
		        + bary_coord[1]*primitives[index].eyeCoords1 \
		        + bary_coord[2]*primitives[index].eyeCoords2;*/

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

__global__ void off_rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

#if defined(BACKFACECULLING)
	  if(primitives[index].toBeDiscard)
		  return;
#endif

	  //get triangle bounding box
	  glm::vec3 boxMin(0);
	  glm::vec3 boxMax(0);
	  getAABBForTriangle(primitives[index],boxMin,boxMax);

	  //get corresponding pixel for bounding box
	  glm::vec2 pixelMin = convertWorldToPixel(boxMin,resolution);
	  glm::vec2 pixelMax = convertWorldToPixel(boxMax,resolution);

	  pixelMin.x = max((int)pixelMin.x,0);
	  pixelMax.x = min((int)pixelMax.x,(int)resolution.x-1);
	  pixelMax.y = max((int)pixelMax.y,0);
	  pixelMin.y = min((int)pixelMin.y,(int)resolution.y-1);

	  fragment frag;
	  int fragIdx = 0;
	  for(int y  = pixelMax.y; y <= pixelMin.y; y++){

		  //loop from xmin to xmax
		  for(int x = pixelMin.x; x <= pixelMax.x; x++) {
			  fragIdx = x + y * resolution.x;

			  //get pixel position in Canonical View Volumes
			  glm::vec2 pixelPoint;
			  pixelPoint.x = (2.0 * x / (float) resolution.x) - 1;
			  pixelPoint.y = 1 - (2.0 * y / (float) resolution.y);

			  //get barycentricCoordinate
			  glm::vec3 barycCoord = calculateBarycentricCoordinate(primitives[index],pixelPoint);

        //check if pixel is within the triangle
			  if(!isBarycentricCoordInBounds(barycCoord)) {
				  continue;
			  }

			  //get depth value
			  float depth = getZAtCoordinate(barycCoord,primitives[index]);

			  //in normalized device coordinate
			  frag.position = glm::vec3(pixelPoint.x,pixelPoint.y,depth);
			  //color interpolation
			  // frag.color = barycCoord.x * primitives[index].c0 + barycCoord.y * primitives[index].c1 + barycCoord.z * primitives[index].c2;
			  // frag.color = mat.diffuseColor;

			  frag.normal = (primitives[index].eyeNormal0 + primitives[index].eyeNormal1
                      + primitives[index].eyeNormal2) / (float)3.0;

			  bool wait = true;
			  while(wait) {
				  if(atomicExch(&(depthbuffer[fragIdx].lock), 1) == 0)
				  {
					  if(depthbuffer[fragIdx].position.x <= -10000 || frag.position.z > depthbuffer[fragIdx].position.z)
					  {
						  depthbuffer[fragIdx] = frag;
					  }
					  depthbuffer[fragIdx].lock = 0;
					  wait = false;
				  }
			  }
		  }
	  }
  }
}

//Implement a fragment shader
//specular lighting, diffuse lighting, ambient lighting
//source opengl-tutoral.org
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPosition, glm::vec3* framebuffer){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	float specular = 400.0;//control width of specular lobe
	float a = 0.1;//ambient
	float d = 0.7;//diffuse
	float s = 0.2;//specular
	fragment frag = depthbuffer[index];

	glm::vec3 lightVector = glm::normalize(lightPosition - frag.position);  // watch out for division by zero
	glm::vec3 normal = glm::normalize(frag.normal); // watch out for division by zero
	float diffuseTerm = glm::clamp(glm::dot(normal, lightVector), 0.0f, 1.0f);

	glm::vec3 R = glm::normalize(reflect(-lightVector, normal)); // watch out for division by zero
	glm::vec3 V = glm::normalize(- frag.position); // watch out for division by zero
      	float specularTerm = pow( fmaxf(glm::dot(R, V), 0.0f), specular );

	framebuffer[index] = a*frag.color + glm::vec3(1.0f) * (d*frag.color*diffuseTerm + s*specularTerm);
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
    printf("piexel render color:\n");
        printVec3(framebuffer[index]);
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 projection, glm::mat4 view, float zNear, float zFar, glm::vec3 lightPosition, float *nbo, int nbosize){
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

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, resolution, projection, view, zNear, zFar, device_vbo_eye, device_nbo, nbosize);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, device_vbo_eye, device_nbo, nbosize, resolution, zNear, zFar);
  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, depth, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec4 light_eyespace = view * glm::vec4(lightPosition, 1.0f);
  lightPosition = glm::vec3(light_eyespace.x, light_eyespace.y, light_eyespace.z);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightPosition, framebuffer);
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
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( device_nbo );
  cudaFree( device_vbo_eye );
  cudaFree( depth );
}
