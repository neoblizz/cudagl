
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

#define BackfaceCulling 1

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;      //screen space vertex buffer
float* device_cbo;      //color buffer(xyz->rgb)
int* device_ibo;        //index buffer
float* device_vbo_eye;  //eye_space vertex buffer
float* device_nbo;      //eye space normal
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
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
 / if(index<vbosize/3){
    //vertex assembly
    glm::vec4 vertex(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1.0f);
    glm::vec4 normal(nbo[3*index], nbo[3*index+1], nbo[3*index+2], 1.0f);
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

#if defined(BackfaceCulling) 
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

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
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
	float specular = 100.0;//control width of specular lobe
	float a = 0.1;//ambient
	float d = 0.1;//diffuse
	float s = 0.1;//specular
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

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  frag.z = -FLT_MAX;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

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
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
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
}

