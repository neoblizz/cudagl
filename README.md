# CUDAGL
CUDA based Graphics Library for NVIDIA's GPUs.

Graphics Processing Units (GPUs) are designed for high throughput, therefore making them really good at compute. NVIDIA's hardware distinguishes the two by using OpenGL for graphics, and CUDA for compute. The OpenGL API is typically used to interact with a GPU, to achieve hardware-accelerated rendering, but CUDA is optimized specifically for a compute mental model. However, systems with limited hardware capabilities like those in consoles or mobile devices use any optimizations available, in some cases, compute is used to do per triangle culling on consoles before data enters the graphics pipeline reducing the overall workload. We plan to take this idea forward by writing an entirely CUDA based graphics pipeline, using a module based code where compute can be called in if it is the right way to go. One of most interesting related work in writing a graphics pipeline stage in CUDA is CUDA Raster, "Unlike previous approaches, we obey ordering constraints imposed by current graphics APIs, guarantee hole-free rasterization, and support multi-sample antialiasing."

## Sources
* S. Laine and T. Karras. High-performance software rasterization on gpus. In Proceedings of the ACM SIGGRAPH Symposium on High Performance Graphics, HPG ’11, pages 79–88, New York, NY, USA, 2011. ACM.
* NVIDIA Corporation. NVIDIA CUDA C programming guide, 2010. Version 3.2.
* The Khronos OpenGL ARB Working Group. Opengl programming guide: The official guide to learning opengl, versions 3.0 and 3.1, 2009.
