
#ifndef GLSLUTILITY_H_
#define GLSLUTILITY_H_

#include <cstdlib>


#include <GL/glew.h>
#include <GLFW/glfw3.h>


namespace glslUtility
{

GLuint createProgram(const char *vertexShaderPath, const char *fragmentShaderPath, const char *attributeLocations[], GLuint numberOfLocations);

}
 
#endif
