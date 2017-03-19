# COMPILERS (CUDA 7.5+)
NVCC     = nvcc -m64
CC       = gcc -m64
CPPC     = g++ -m64


# COMPILER FLAGS
CUDA_FLAGS = -I/usr/local/cuda/samples/common/inc -I/usr/local/cuda/include
LFLAGS = -lglut -lGL -lGLEW

# INCLUDES
SOURCE = ../src/render.cu ../src/render.h ../src/tools.h ../src/utilities.h ../src/cudaMat4.h ../src/cudaError.h
PIPELINE = ../src/pipeline/vertexshader.cu ../src/pipeline/primitiveassembly.cu ../src/pipeline/rasterizer.cu ../src/pipeline/fragmentshader.cu
GLSL = ../src/glslUtility.cpp ../src/glslUtility.h
OBJ = ../src/ObjCore/obj.cpp
OBJL = ../src/ObjCore/objloader.cpp

# MAKE ALL
all: cudagl

render.o: $(SOURCE) $(PIPELINE)
	$(NVCC) $(CUDA_FLAGS) -c ../src/render.cu

glslUtility.o: $(GLSL)
	$(CPPC) $(LFLAGS) $(CUDA_FLAGS) -c ../src/glslUtility.cpp

utilities.o: ../src/utilities.cpp ../src/utilities.h ../src/cudaMat4.h
	$(CPPC) $(CUDA_FLAGS) -c ../src/utilities.cpp

main.o: ../src/main.cpp ../src/main.h ../src/render.h ../src/utilities.h ../src/cudaMat4.h
	$(CPPC) $(LFLAGS) $(CUDA_FLAGS) -c ../src/main.cpp

obj.o: $(OBJ)
	$(CPPC) -c $(OBJ)

objloader.o: $(OBJL)
	$(CPPC) -c $(OBJL)

cudagl: main.o render.o glslUtility.o utilities.o obj.o objloader.o
	$(NVCC) $(LFLAGS) $(CUDA_FLAGS) *.o -o cudagl

# MAKE CLEAN
clean:
	rm *.o
	rm cudagl

# MAKE TEST
test: cudagl
	./cudagl mesh=../src/cow.obj
