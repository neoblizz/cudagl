#ifndef FRAGMENT_SHADER_H
#define FRAGMENT_SHADER_H

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

#endif // FRAGMENT_SHADER_H
