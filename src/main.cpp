///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2017, STEREOLABS.
//
// All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

/*************************************************************************
** This sample shows how to capture a real-time 3D reconstruction      **
** of the scene using the Spatial Mapping API. The resulting mesh      **
** is displayed as a wireframe on top of the left image using OpenGL.  **
** Spatial Mapping can be started and stopped with the Space Bar key   **
*************************************************************************/

// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// OpenGL includes
#include <GL/glew.h>
//#include <GL/freeglut.h>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include <GLObject.hpp>
#include "utils.hpp"
#include <cuda_gl_interop.h>

//#include <opencv.hpp>

// Define if you want to use the mesh as a set of chunks or as a global entity.
#define USE_CHUNKS 1

// ZED object (camera, mesh, pose)
sl::Camera zed;
sl::Mat left_image; // sl::Mat to hold images
sl::Pose pose;      // sl::Pose to hold pose data
sl::Mesh mesh;      // sl::Mesh to hold the mesh generated during spatial mapping
sl::SpatialMappingParameters spatial_mapping_params;
sl::MeshFilterParameters filter_params;
sl::TRACKING_STATE tracking_state;
sl::Mat point_cloud;

// For CUDA-OpenGL interoperability
cudaGraphicsResource* cuda_gl_ressource;//cuda GL resource           

										// OpenGL mesh container
//std::vector<MeshObject> mesh_object;    // Opengl mesh container
sl::float3 vertices_color;              // Defines the color of the mesh

										// OpenGL camera projection matrix
sl::Transform camera_projection;

// Opengl object
// Shader* shader_mesh = NULL; //GLSL Shader for mesh
// Shader* shader_image = NULL;//GLSL Shader for image
GLuint imageTex;            //OpenGL texture mapped with a cuda array (opengl gpu interop)
GLuint shMVPMatrixLoc;      //Shader variable loc
GLuint shColorLoc;          //Shader variable loc
GLuint texID;               //Shader variable loc (sampler/texture)
GLuint fbo = 0;             //FBO
GLuint renderedTexture = 0; //Render Texture for FBO
GLuint quad_vb;             //buffer for vertices/coords for image

							// OpenGL Viewport size

// Spatial Mapping status
bool mapping_is_started = false;
std::chrono::high_resolution_clock::time_point t_last;

//// Sample functions
void close();
void run();
void startMapping();
void stopMapping();
void keyPressedCallback(unsigned char c, int x, int y);

int main(int argc, char** argv) {
	// Init GLUT window
// 	glutInit(&argc, argv);
// 	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	// Setup configuration parameters for the ZED    
	sl::InitParameters parameters;
	if (argc > 1) parameters.svo_input_filename = argv[1];

	parameters.depth_mode = sl::DEPTH_MODE_QUALITY; // Use QUALITY depth mode to improve mapping results
	parameters.coordinate_units = sl::UNIT_METER;
	parameters.coordinate_system = sl::COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP; // OpenGL coordinates system

																			// Open the ZED
	sl::ERROR_CODE err = zed.open(parameters);
	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cout << sl::toString(err) << std::endl;
		zed.close();
		return -1;
	}

	// Configure Spatial Mapping and filtering parameters
	spatial_mapping_params.range_meter = sl::SpatialMappingParameters::get(sl::SpatialMappingParameters::MAPPING_RANGE_FAR);
	spatial_mapping_params.resolution_meter = 1;// sl::SpatialMappingParameters::get(sl::SpatialMappingParameters::MAPPING_RESOLUTION_MEDIUM);
	spatial_mapping_params.save_texture = true;
	spatial_mapping_params.max_memory_usage = 4096;
	spatial_mapping_params.use_chunk_only = USE_CHUNKS; // If we use chunks we do not need to keep the mesh consistent
	
	filter_params.set(sl::MeshFilterParameters::MESH_FILTER_MEDIUM);

	// Initialize OpenGL
// 	int res = initGL();
// 	if (res != 0) {
// 		std::cout << "Failed to initialize OpenGL" << std::endl;
// 		zed.close();
// 		return -1;
// 	}

	std::cout << "*************************************************************" << std::endl;
	std::cout << "**      Press the Space Bar key to start and stop          **" << std::endl;
	std::cout << "*************************************************************" << std::endl;

	// Set glut callback before start
// 	glutKeyboardFunc(keyPressedCallback);// Callback that starts spatial mapping when space bar is pressed
// 	glutDisplayFunc(run); // Callback that updates mesh data
// 	glutCloseFunc(close);// Close callback
// 
// 						 // Start the glut main loop thread
// 	glutMainLoop();
	sl::RuntimeParameters rt_parameters = sl::RuntimeParameters();
	rt_parameters.sensing_mode = sl::SENSING_MODE_FILL;
	rt_parameters.enable_point_cloud = true;
	if (zed.grab(rt_parameters) == sl::SUCCESS) {

		zed.retrieveMeasure(point_cloud, sl::MEASURE_XYZRGBA);
		std::string filename = getDir() + "pc.ply";
		// 	sl::ERROR_CODE ec = point_cloud.write(filename.c_str());
		// 	std::cout << ">> Point cloud has been saved under pc.ply with error code:" << ec << std::endl;

		if (sl::savePointCloudAs(point_cloud, sl::POINT_CLOUD_FORMAT_PLY_ASCII, filename.c_str(), true))
		{
			std::cout << ">> Point cloud has been saved" << std::endl;
		}
		else
		{
			std::cout << ">> Point cloud has been not saved" << std::endl;
		}

	}


	return 0;
}

/**
This function close the sample (when a close event is generated)
**/
void close() {
	left_image.free();

// 	if (shader_mesh) delete shader_mesh;
// 	if (shader_image) delete shader_image;

	//mesh_object.clear();
	zed.close();
}

/**
Start the spatial mapping process
**/
void startMapping() {
	// clear previously used objects
	mesh.clear();
	//mesh_object.clear();

#if !USE_CHUNKS
	// Create only one object that will contain the full mesh.
	// Otherwise, different MeshObject will be created for each chunk when needed
	mesh_object.emplace_back();
#endif

	// Enable positional tracking before starting spatial mapping
	zed.enableTracking();
	// Enable spatial mapping
	zed.enableSpatialMapping(spatial_mapping_params);

	// Start a timer, we retrieve the mesh every XXms.
	t_last = std::chrono::high_resolution_clock::now();

	mapping_is_started = true;
	std::cout << "** Spatial Mapping is started ... **" << std::endl;
}

/**
Stop the spatial mapping process
**/
void stopMapping() {
	// Stop the mesh request and extract the whole mesh to filter it and save it as an obj file
	mapping_is_started = false;
	std::cout << "** Stop Spatial Mapping ... **" << std::endl;

// 	// Extract the whole mesh
// 	sl::Mesh wholeMesh;
// 	zed.extractWholeMesh(wholeMesh);
// 	std::cout << ">> Mesh has been extracted..." << std::endl;
// 
// 	// Filter the extracted mesh
// 	wholeMesh.filter(filter_params, USE_CHUNKS);
// 	std::cout << ">> Mesh has been filtered..." << std::endl;
// 
// 	// If textures have been saved during spatial mapping, apply them to the mesh
// 	if (spatial_mapping_params.save_texture) {
// 		wholeMesh.applyTexture(sl::MESH_TEXTURE_RGB);
// 		std::cout << ">> Mesh has been textured..." << std::endl;
// 	}
// 
// 	//Save as an OBJ file
// 	std::string saveName = getDir() + "mesh_gen.obj";
// 	bool t = wholeMesh.save(saveName.c_str());
// 	if (t) std::cout << ">> Mesh has been saved under " << saveName << std::endl;
// 	else std::cout << ">> Failed to save the mesh under  " << saveName << std::endl;

	// Update the displayed Mesh
// #if USE_CHUNKS
// 	mesh_object.clear();
// 	mesh_object.resize(wholeMesh.chunks.size());
// 	for (int c = 0; c < wholeMesh.chunks.size(); c++)
// 		mesh_object[c].updateMesh(wholeMesh.chunks[c].vertices, wholeMesh.chunks[c].triangles);
// #else
// 	mesh_object[0].updateMesh(wholeMesh.vertices, wholeMesh.triangles);
// #endif

	zed.retrieveMeasure(point_cloud, sl::MEASURE_XYZRGBA);
	std::string filename = getDir() + "pc.ply";
// 	sl::ERROR_CODE ec = point_cloud.write(filename.c_str());
// 	std::cout << ">> Point cloud has been saved under pc.ply with error code:" << ec << std::endl;

	if (sl::savePointCloudAs(point_cloud, sl::POINT_CLOUD_FORMAT_PLY_ASCII, filename.c_str(), true))
	{
		std::cout << ">> Point cloud has been saved" << std::endl;
	}
	else
	{
		std::cout << ">> Point cloud has been not saved" << std::endl;
	}
}

void pauseMapping()
{
	zed.pauseSpatialMapping(true);
}

void resumeMapping()
{
	t_last = std::chrono::high_resolution_clock::now();
	zed.pauseSpatialMapping(false);
}

/**
Update the mesh and draw image and wireframe using OpenGL
**/
void run() {
	sl::RuntimeParameters rt_parameters = sl::RuntimeParameters();
	rt_parameters.sensing_mode = sl::SENSING_MODE_FILL;
	rt_parameters.enable_point_cloud = true;
	if (zed.grab(rt_parameters) == sl::SUCCESS) {
		// Retrieve image in GPU memory
		zed.retrieveImage(left_image, sl::VIEW_LEFT, sl::MEM_GPU);

		// CUDA - OpenGL interop : copy the GPU buffer to a CUDA array mapped to the texture.
		cudaArray_t ArrIm;
		cudaGraphicsMapResources(1, &cuda_gl_ressource, 0);
		cudaGraphicsSubResourceGetMappedArray(&ArrIm, cuda_gl_ressource, 0, 0);
		cudaMemcpy2DToArray(ArrIm, 0, 0, left_image.getPtr<sl::uchar1>(sl::MEM_GPU), left_image.getStepBytes(sl::MEM_GPU), left_image.getPixelBytes()*left_image.getWidth(), left_image.getHeight(), cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &cuda_gl_ressource, 0);

		// Update pose data (used for projection of the mesh over the current image)
		tracking_state = zed.getPosition(pose);

		if (mapping_is_started) {

			// Compute elapse time since the last call of sl::Camera::requestMeshAsync()
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t_last).count();
			// Ask for a mesh update if 500ms have spend since last request
			if (duration > 500) {
				zed.requestMeshAsync();
				t_last = std::chrono::high_resolution_clock::now();
			}

			if (zed.getMeshRequestStatusAsync() == sl::SUCCESS) {
				// Get the current mesh generated and send it to opengl
				if (zed.retrieveMeshAsync(mesh) == sl::SUCCESS) {
#if USE_CHUNKS
					for (int c = 0; c < mesh.chunks.size(); c++) {
						// If the chunk does not exist in the rendering process -> add it in the rendering list
						//if (mesh_object.size() < mesh.chunks.size()) mesh_object.emplace_back();
						// If the chunck has been updated by the spatial mapping, update it for rendering
// 						if (mesh.chunks[c].has_been_updated)
// 							mesh_object[c].updateMesh(mesh.chunks[c].vertices, mesh.chunks[c].triangles);
					}
#else
					//mesh_object[0].updateMesh(mesh.vertices, mesh.triangles);
#endif
				}
			}
		}

		// Display image and mesh using OpenGL 
		//drawGL();
	}

	// If SVO input is enabled, close the window and stop mapping if video reached the end
// 	if (zed.getSVOPosition() > 0 && zed.getSVOPosition() == zed.getSVONumberOfFrames() - 1)
// 		glutLeaveMainLoop();
// 
// 	// Prepare next update
// 	glutPostRedisplay();
}

/**
This function handles keyboard events (especially space bar to start the mapping)
**/
void keyPressedCallback(unsigned char c, int x, int y) {
	switch (c) {
	case 32: // Space bar id	
		if (!mapping_is_started) // User press the space bar and spatial mapping is not started 
			startMapping();
		else // User press the space bar and spatial mapping is started 
			stopMapping();
		break;
	case 'q':
		//glutLeaveMainLoop(); // End the process	
		break;

	case 'p':
		pauseMapping();
		break;
	case 'r':
		resumeMapping();
		break;
	default:
		break;
	}
}
