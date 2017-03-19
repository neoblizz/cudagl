#ifndef PRIMITIVE_ASSEMBLY_H
#define PRIMITIVE_ASSEMBLY_H

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

#endif // PRIMITIVE_ASSEMBLY_H
