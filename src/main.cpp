#include "main.h"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

// Camera
glm::mat4 cam = glm::mat4( 1.0 ); // eye
// Projection matrix
glm::mat4 projection = glm::perspective(60.0f, 1.0f, 0.1f, 100.0f);
// Light
glm::vec3 light( 1.0, 1.0, 1.0 );
// Drawing mode
int draw_mode = DRAW_COLOR;

// Mouse interactivity
int mouse_left_down = false;
int mouse_right_down = false;
int mouse_in_down = false;
glm::vec2 prev_xy;


int main(int argc, char** argv){

  bool loadedScene = false;
  for(int i=1; i<argc; i++){
    string header; string data;
    istringstream liness(argv[i]);
    getline(liness, header, '='); getline(liness, data, '=');
    if(strcmp(header.c_str(), "mesh")==0){
      //renderScene = new scene(data);
      mesh = new obj();
      objLoader* loader = new objLoader(data, mesh);
      mesh->buildVBOs();
      delete loader;
      loadedScene = true;
    }
  }

  if(!loadedScene){
    cout << "Usage: mesh=[obj file]" << endl;
    return 0;
  }

  frame = 0;
  seconds = time (NULL);
  fpstracker = 0;

  // Launch CUDA/GL
  #ifdef __APPLE__
  // Needed in OSX to force use of OpenGL3.2
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
  glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
  glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  init();
  #else
  init(argc, argv);
  #endif

  // Initialize camera position
  cam = glm::translate( cam, glm::vec3( 0.0, 0.0, 2.0f ) );

  initCuda();

  initVAO();
  initTextures();

  GLuint passthroughProgram;
  passthroughProgram = initShader("shaders/passthroughVS.glsl", "shaders/passthroughFS.glsl");

  glUseProgram(passthroughProgram);
  glActiveTexture(GL_TEXTURE0);

  #ifdef __APPLE__
    // send into GLFW main loop
    while(1){
      display();
      if (glfwGetKey(GLFW_KEY_ESC) == GLFW_PRESS || !glfwGetWindowParam( GLFW_OPENED )){
          kernelCleanup();
          cudaDeviceReset();
          exit(0);
      }
    }

    glfwTerminate();
  #else
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(mouse_motion);

    glutMainLoop();
  #endif
  kernelCleanup();
  return 0;
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda(){
  // Map OpenGL buffer object for writing from CUDA on a single GPU
  // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
  dptr=NULL;

  vbo = mesh->getVBO();
  vbosize = mesh->getVBOsize();

  float newcbo[] = {0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                    1.0, 0.0, 0.0};
  cbo = newcbo;
  cbosize = 9;

  ibo = mesh->getIBO();
  ibosize = mesh->getIBOsize();

  nbo = mesh->getNBO();
  nbosize = mesh->getNBOsize();

  cudaGLMapBufferObject((void**)&dptr, pbo);

  cudaRasterizeCore(glm::inverse(cam), projection, light, draw_mode, dptr, glm::vec2(width, height), frame, vbo, vbosize, nbo, nbosize, cbo, cbosize, ibo, ibosize);
  cudaGLUnmapBufferObject(pbo);

  vbo = NULL;
  cbo = NULL;
  ibo = NULL;

  frame++;
  fpstracker++;

}

void display(){

  // DEBUG: display only one frame
  /*
  if ( frame > 5 )
    return;
  */
  runCuda();
time_t seconds2 = time (NULL);

  if(seconds2-seconds >= 1){

    fps = fpstracker/(seconds2-seconds);
    fpstracker = 0;
    seconds = seconds2;

  }

  string title = " Rasterizer | "+ utilityCore::convertIntToString((int)fps) + "FPS";
  glutSetWindowTitle(title.c_str());

  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
  glBindTexture(GL_TEXTURE_2D, displayImage);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
      GL_RGBA, GL_UNSIGNED_BYTE, NULL);

  glClear(GL_COLOR_BUFFER_BIT);

  // VAO, shader program, and texture already bound
  glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

  glutPostRedisplay();
  glutSwapBuffers();
}

void mouse_motion( int x, int y ) {
  glm::vec2 current_xy;
  glm::vec2 dxy;
  if ( mouse_left_down ) {
    current_xy = glm::vec2(x,y);
    dxy = current_xy - prev_xy;
    prev_xy = current_xy;

    printf(" dxy: [%f, %f] \n", dxy.x, dxy.y );
    cam = glm::rotate( cam, -dxy.x/5.0f, glm::vec3(0.0, 1.0, 0.0));
    cam = glm::rotate( cam, dxy.y/5.0f, glm::vec3(1.0, 0.0, 0.0));
  }
  if ( mouse_right_down ) {
    current_xy = glm::vec2(x,y);
    dxy = current_xy - prev_xy;
    prev_xy = current_xy;

    printf(" dxy: [%f, %f] \n", dxy.x, dxy.y );
    cam = glm::translate( cam, glm::vec3(-4.0*dxy.x/width, 0.0, 0.0));
    cam = glm::translate( cam, glm::vec3(0.0, -4.0*dxy.y/height, 0.0));
  }
 if ( mouse_in_down ) {
    current_xy = glm::vec2(x,y);
    dxy = current_xy - prev_xy;
    prev_xy = current_xy;

    printf(" dxy: [%f, %f] \n", dxy.x, dxy.y );
    cam = glm::translate( cam, glm::vec3(0.0, 0.0, 5.0*dxy.y/width));
    cam = glm::rotate( cam, dxy.x/5.0f, glm::vec3(0.0, 0.0, 1.0));

 }

}

// Mouse interactive camera
void mouse( int button, int state, int x, int y ) {
  int modifier = glutGetModifiers();
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && modifier != GLUT_ACTIVE_SHIFT && modifier != GLUT_ACTIVE_CTRL) {
    printf( "mouse_down: true \n" );
    mouse_left_down = true;
    prev_xy = glm::vec2( x, y );
  }
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP ) {
    printf( "mouse_down: false \n" );
    mouse_left_down = false;
  }
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && modifier == GLUT_ACTIVE_SHIFT && modifier != GLUT_ACTIVE_CTRL ) {
    printf( "mouse_down: true \n" );
    mouse_right_down = true;
    prev_xy = glm::vec2( x, y );
  }
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP ) {
    printf( "mouse_down: false \n" );
    mouse_right_down = false;
  }
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN && modifier != GLUT_ACTIVE_SHIFT && modifier == GLUT_ACTIVE_CTRL ) {
    printf( "mouse_down: true \n" );
    mouse_in_down = true;
    prev_xy = glm::vec2( x, y );
  }
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP ) {
    printf( "mouse_down: false \n" );
    mouse_in_down = false;
  }
}



void keyboard(unsigned char key, int x, int y)
{
  switch (key)
  {
     case(27):
       shut_down(1);
       break;
     // Rotate camera
     case('i'):
 cam = glm::rotate( cam, 10.0f, glm::vec3( 1.0, 0.0, 0.0 ) );
 break;
     case('k'):
 cam = glm::rotate( cam, -10.0f, glm::vec3( 1.0, 0.0, 0.0 ) );
 break;
     case('j'):
 cam = glm::rotate( cam, 10.0f, glm::vec3( 0.0, 1.0, 0.0 ) );
 break;
     case('l'):
 cam = glm::rotate( cam, -10.0f, glm::vec3( 0.0, 1.0, 0.0 ) );
 break;
    // Translate camera
    case('w'):
 cam = glm::translate( cam, glm::vec3( 0.0, 0.0, -0.1 ) );
 break;
    case('s'):
 cam = glm::translate( cam, glm::vec3( 0.0, 0.0, 0.1) );
 break;
    case('a'):
 cam = glm::translate( cam, glm::vec3( 0.1, 0.0, 0.0 ) );
 break;
    case('d'):
 cam = glm::translate( cam, glm::vec3( -0.1, 0.0, 0.0 ) );
 break;
    case('1'):
 draw_mode = DRAW_SOLID;
 break;
    case('2'):
 draw_mode = DRAW_COLOR;
 break;
    case('3'):
 draw_mode = DRAW_NORMAL;
 break;
    case('4'):
 draw_mode = SHADE_SOLID;
 break;
    case('5'):
 draw_mode = SHADE_COLOR;
 break;
  }
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------


void init(int argc, char* argv[]){
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow("Rasterizer");

  // Init GLEW
  glewInit();
  GLenum err = glewInit();
  if (GLEW_OK != err)
  {
    /* Problem: glewInit failed, something is seriously wrong. */
    std::cout << "glewInit failed, aborting." << std::endl;
    exit (1);
  }

  initVAO();
  initTextures();
}

void initPBO(GLuint* pbo){
  if (pbo) {
    // set up vertex data parameter
    int num_texels = width*height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void initCuda(){
  // Use device with highest Gflops/s
  cudaGLSetGLDevice( compat_getMaxGflopsDeviceId() );

  initPBO(&pbo);

  // Clean up on program exit
  atexit(cleanupCuda);

  runCuda();
}

void initTextures(){
    glGenTextures(1,&displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
        GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void){
    GLfloat vertices[] =
    {
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] =
    {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader(const char *vertexShaderPath, const char *fragmentShaderPath){
    GLuint program = glslUtility::createProgram(vertexShaderPath, fragmentShaderPath, attributeLocations, 2);
    GLint location;

    glUseProgram(program);

    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda(){
  if(pbo) deletePBO(&pbo);
  if(displayImage) deleteTexture(&displayImage);
}

void deletePBO(GLuint* pbo){
  if (pbo) {
    // unregister this buffer object with CUDA
    cudaGLUnregisterBufferObject(*pbo);

    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);

    *pbo = (GLuint)NULL;
  }
}

void deleteTexture(GLuint* tex){
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code){
  kernelCleanup();
  cudaDeviceReset();
  #ifdef __APPLE__
  glfwTerminate();
  #endif
  exit(return_code);
}
