#All cuda paths are for v5.0+

NVCC     = nvcc -m64
CC       = gcc -m64
CPPC     = g++ -m64


CUDA_FLAGS = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include

# LFLAGS = -lGLEW -lglfw3 -lGL -lm -ldl -lX11 -lXxf86vm -lXrandr -lpthread -lXi -lXcursor -lXinerama
LFLAGS = -lglut -lGL -lGLEW

all: cudagl

rasterizeKernels.o: ../src/render.cu ../src/render.h ../src/tools.h ../src/utilities.h ../src/cudaMat4.h ../src/cudaError.h
	$(NVCC) $(CUDA_FLAGS) -c ../src/render.cu

glslUtility.o: ../src/glslUtility.cpp ../src/glslUtility.h
	$(CPPC) $(LFLAGS) $(CUDA_FLAGS) -c ../src/glslUtility.cpp

utilities.o: ../src/utilities.cpp ../src/utilities.h ../src/cudaMat4.h
	$(CPPC) $(CUDA_FLAGS) -c ../src/utilities.cpp

main.o: ../src/main.cpp ../src/main.h ../src/rasterizeKernels.h ../src/utilities.h ../src/cudaMat4.h
	$(CPPC) $(LFLAGS) $(CUDA_FLAGS) -c ../src/main.cpp

obj.o: ../src/ObjCore/obj.cpp
	$(CPPC) -c ../src/ObjCore/obj.cpp

objloader.o: ../src/ObjCore/objloader.cpp
	$(CPPC) -c ../src/ObjCore/objloader.cpp

cudagl: main.o rasterizeKernels.o glslUtility.o utilities.o obj.o objloader.o
	$(NVCC) $(LFLAGS) $(CUDA_FLAGS) *.o -o cudagl


clean:
	rm *.o
	rm cudagl

test: cudagl
	./cudagl
