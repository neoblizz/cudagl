#ifndef RASTERIZER_H
#define RASTERIZER_H

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


#endif // RASTERIZER_H
