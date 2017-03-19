#ifndef VERTEX_SHADER_H
#define VERTEX_SHADER_H

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

#endif // VERTEX_SHADER_H
