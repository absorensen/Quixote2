// based on implementations by 
// Joey de Vries: https://learnopengl.com/code_viewer_gh.php?code=src/5.advanced_lighting/9.ssao/ssao.cpp
// Joshua Senouf: https://github.com/JoshuaSenouf/GLEngine/blob/master/src/renderer/glengine.cpp
// Kevin Fung: https://github.com/Polytonic/Glitter

// Local Headers
#include "quixote.hpp"

// settings
const unsigned int SCR_WIDTH = 512;
const unsigned int SCR_HEIGHT = 512;
const float ASPECT_RATIO = ((float)SCR_WIDTH) / SCR_HEIGHT;
static const float scale = 1.0f / static_cast<float>(SCR_WIDTH);

// camera
Camera camera(-2.0f, 1.5f, 3.4f, 0.0f, 1.0f, 0.0f, -50.0f, -12.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
double deltaTime = 0.0f;
double lastFrame = 0.0;



// deferred
unsigned int gBuffer, gPosition, gNormal, gAlbedoSpec, gAO, accumulatorDepth, deferredOutput;

// ssao
unsigned int ssaoFBO, ssaoBlurFBO, ssaoBuffer, ssaoBufferBlur, noiseTexture;
std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0); // generates random floats between 0.0 and 1.0
std::default_random_engine generator;
std::vector<glm::vec3> ssaoKernel;
std::vector<glm::vec3> ssaoNoise;

// forward
unsigned int forwardFBO, forwardBuffer, forwardDepth, forwardOutput;


// switch between primary and laplacian domain post processing pipelines
bool laplace_pipeline = true;

// post-process
unsigned int postProcessFBO, postProcessBuffer, postProcessDepth, postProcessOutput;
bool post_process_edges = true;
bool post_process_transparency = true;

// reconstruction
unsigned int laplacePostProcessFBO, laplaceProcessBuffer, laplacePostProcessDepth, laplacePostProcessOutput, laplaceReconstructionOutput;
bool laplace_edges = true;
bool laplace_transparency = true;
//unsigned int source_list;
//Shader g2r_prog;

// output stage
glm::vec3 gamma(1.0f / 2.2f);
float exposure = 1.0f;
bool uncharted_tonemap = true;
bool print_timing = true;

// object specific
// ---------------
// lights
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
const unsigned int NR_LIGHTS = 16;
std::vector<glm::vec3> lightPositions;
std::vector<glm::vec3> lightColors;
const float constant = 1.0f;
const float linear = 0.7f;
const float quadratic = 1.8f;
const glm::vec3 point_lights_size(0.0175f);

// nanosuits
std::vector<glm::vec3> objectPositions;

// screen quads
unsigned int quadVAO = 0;
unsigned int quadVBO;





int main(int argc, char * argv[]) {

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_SAMPLES, 8);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Quixote", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	// configure global opengl state
	// -----------------------------
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	Reconstructor* reconstructor = new Reconstructor(SCR_WIDTH, SCR_HEIGHT, 1, GL_RGBA, CONV_PYR_CPU);

	std::string path("Glitter/Sources/");
	Shader shaderGeometryPass(path+"g_buffer.vert", path + "g_buffer.frag");
	Shader shaderLightingPass(path + "simple.vert", path + "deferred_shading.frag");
	Shader shaderSSAO(path + "simple.vert", path + "ssao.frag");
	Shader shaderSSAOBlur(path + "simple.vert", path + "ssao_blur.frag");
	Shader shaderForward(path + "forward.vert", path + "forward.frag");

	// postprocessing path 1 - if reconstruction==false
	Shader shaderPostProcess(path + "simple.vert", path + "post_process.frag");
	Shader shaderOutput(path + "simple.vert", path + "output.frag");
	
	// postprocessing path 2 - if reconstruction==true
	Shader shaderLaplacePostProcess(path + "simple.vert", path + "laplace_post_process.frag");
	Shader shaderLaplaceReconstructionOutput(path + "simple.vert", path + "laplace_reconstruction_output.frag");
	
	//g2r_prog = Shader(path+ "fft_texcoord.vert", path + "fft_g2r.frag");

	// -- making models, making shit, fighting round the world --
	Model nanosuit("Glitter/Resources/models/nanosuit/nanosuit.obj", true);
	const float offset_models = 2.0;
	objectPositions.push_back(glm::vec3(-offset_models, -offset_models, -offset_models));
	objectPositions.push_back(glm::vec3(0.0, -offset_models, -offset_models));
	objectPositions.push_back(glm::vec3(offset_models, -offset_models, -offset_models));
	objectPositions.push_back(glm::vec3(-offset_models, -offset_models, 0.0));
	objectPositions.push_back(glm::vec3(0.0, -offset_models, 0.0));
	objectPositions.push_back(glm::vec3(offset_models, -offset_models, 0.0));
	objectPositions.push_back(glm::vec3(-offset_models, -offset_models, 3.0));
	objectPositions.push_back(glm::vec3(0.0, -offset_models, offset_models));
	objectPositions.push_back(glm::vec3(offset_models, -offset_models, offset_models));

	// -- deferred rendering --
	glGenFramebuffers(1, &gBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
	// position color buffer
	glGenTextures(1, &gPosition);
	glBindTexture(GL_TEXTURE_2D, gPosition);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
	// normal color buffer
	glGenTextures(1, &gNormal);
	glBindTexture(GL_TEXTURE_2D, gNormal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
	// color + specular color buffer
	glGenTextures(1, &gAlbedoSpec);
	glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);
	// ambient occlusion buffer
	glGenTextures(1, &gAO);
	glBindTexture(GL_TEXTURE_2D, gAO);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, gAO, 0);
	// accumulator for postProcessing
	glGenTextures(1, &deferredOutput);
	glBindTexture(GL_TEXTURE_2D, deferredOutput);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, deferredOutput, 0);

	// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
	GLuint deferredAttachments[5] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3, GL_COLOR_ATTACHMENT4 };
	glDrawBuffers(5, deferredAttachments);
	
	
	
	// create and attach depth buffer (renderbuffer)
	glGenRenderbuffers(1, &accumulatorDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, accumulatorDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, accumulatorDepth);
	// finally check if framebuffer is complete
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);




	// -- SSAO --
	glGenFramebuffers(1, &ssaoFBO);  glGenFramebuffers(1, &ssaoBlurFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
	// SSAO color buffer
	glGenTextures(1, &ssaoBuffer);
	glBindTexture(GL_TEXTURE_2D, ssaoBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoBuffer, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Framebuffer not complete!" << std::endl;
	// and blur stage
	glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
	glGenTextures(1, &ssaoBufferBlur);
	glBindTexture(GL_TEXTURE_2D, ssaoBufferBlur);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssaoBufferBlur, 0);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
	glBindFramebuffer(GL_FRAMEBUFFER, 0);




	// generate sample kernel
	// ----------------------

	for (unsigned int i = 0; i < 64; ++i)
	{
		glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
		sample = glm::normalize(sample);
		sample *= randomFloats(generator);
		float scale = float(i) / 64.0f;

		// scale samples s.t. they're more aligned to center of kernel
		scale = lerp(0.1f, 1.0f, scale * scale);
		sample *= scale;
		ssaoKernel.push_back(sample);
	}

	// generate noise texture
	// ----------------------
	for (unsigned int i = 0; i < 16; i++)
	{
		glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f); // rotate around z-axis (in tangent space)
		ssaoNoise.push_back(noise);
	}
	glGenTextures(1, &noiseTexture);
	glBindTexture(GL_TEXTURE_2D, noiseTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssaoNoise[0]);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);


	// setup lightsBuffer
	glGenFramebuffers(1, &forwardFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, forwardFBO);

	//glGenTextures(1, &forwardBuffer);
	//glBindTexture(GL_TEXTURE_2D, forwardBuffer);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, forwardBuffer, 0);

	glGenTextures(1, &forwardOutput);
	glBindTexture(GL_TEXTURE_2D, forwardOutput);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, forwardOutput, 0);

	glGenRenderbuffers(1, &forwardDepth);
	glBindRenderbuffer(GL_RENDERBUFFER, forwardDepth);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, forwardDepth);

	GLuint attachmentsForward[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachmentsForward);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;



	// setup post process
	// ------------------
	// first post-processing buffer
	glGenFramebuffers(1, &postProcessFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);

	glGenTextures(1, &postProcessOutput);
	glBindTexture(GL_TEXTURE_2D, postProcessOutput);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, postProcessOutput, 0);
	GLuint attachmentsPostProcess[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachmentsPostProcess);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	// setup reconstruction post processing
	// ------------------------------------
	glGenFramebuffers(1, &laplacePostProcessFBO);
	glBindFramebuffer(GL_FRAMEBUFFER, laplacePostProcessFBO);

	glGenTextures(1, &laplacePostProcessOutput);
	glBindTexture(GL_TEXTURE_2D, laplacePostProcessOutput);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, laplacePostProcessOutput, 0);

	GLuint attachmentsReconstruction[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, attachmentsReconstruction);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "Framebuffer not complete!" << std::endl;

	// reconstruction - fft
	//FFT* fft = new FFT(SCR_WIDTH, SCR_HEIGHT, reconstructionPostProcessOutput);
	//FFT* fft = new FFT(SCR_WIDTH, SCR_HEIGHT);

	glGenTextures(1, &laplaceReconstructionOutput);
	glBindTexture(GL_TEXTURE_2D, laplaceReconstructionOutput);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// lighting info
	// -------------
	srand(16);
	for (unsigned int i = 0; i < NR_LIGHTS; i++)
	{
		// calculate slightly random offsets
		float xPos = ((rand() % 100) / 100.0f) * 6.0f - 3.0f;
		float yPos = ((rand() % 100) / 100.0f) * 6.0f - 4.0f;
		float zPos = ((rand() % 100) / 100.0f) * 6.0f - 3.0f;
		lightPositions.push_back(glm::vec3(xPos, yPos, zPos));
		// also calculate random color
		float rColor = 2.0f * ((rand() % 100) / 200.0f) + 0.5f; // between 0.5 and 1.0
		float gColor = 2.0f * ((rand() % 100) / 200.0f) + 0.5f; // between 0.5 and 1.0
		float bColor = 2.0f * ((rand() % 100) / 200.0f) + 0.5f; // between 0.5 and 1.0
		lightColors.push_back(glm::vec3(rColor, gColor, bColor));
	}
	lightColors[2] *= 5.0f;
	lightColors[5] *= 20.0f;
	lightColors[9] *= 30.0f;
	lightColors[15] *= 60.0f;


	// shader configuration
	// --------------------
	shaderLightingPass.use();
	shaderLightingPass.setInt("gPosition", 0);
	shaderLightingPass.setInt("gNormal", 1);
	shaderLightingPass.setInt("gAlbedoSpec", 2);
	shaderLightingPass.setInt("gAO", 3);

	shaderSSAO.use();
	shaderSSAO.setInt("gPosition", 0);
	shaderSSAO.setInt("gNormal", 1);
	shaderSSAO.setInt("texNoise", 2);

	shaderSSAOBlur.use();
	shaderSSAOBlur.setInt("ssaoInput", 0);
	
	//if (reconstruction) {
		shaderLaplacePostProcess.use();
		shaderLaplacePostProcess.setInt("deferredOutput", 0);
		shaderLaplacePostProcess.setInt("forwardOutput", 1);
		shaderLaplacePostProcess.setBool("edges", laplace_edges);
		shaderLaplacePostProcess.setBool("transparency", laplace_transparency);

		shaderLaplaceReconstructionOutput.use();
		shaderLaplaceReconstructionOutput.setInt("reconstruction_output", 0);
		shaderLaplaceReconstructionOutput.setFloat("exposure", exposure);
		shaderLaplaceReconstructionOutput.setVec3("gamma", gamma);
		shaderLaplaceReconstructionOutput.setBool("uncharted_tonemap", uncharted_tonemap);
	/*}
	else {*/
		shaderPostProcess.use();
		shaderPostProcess.setInt("deferredOutput", 0);
		shaderPostProcess.setInt("forwardOutput", 1);
		shaderPostProcess.setBool("edges", post_process_edges);
		shaderPostProcess.setBool("transparency", post_process_transparency);

		shaderOutput.use();
		shaderOutput.setInt("postProcessOutput", 0);
		shaderOutput.setFloat("exposure", exposure);
		shaderOutput.setVec3("gamma", gamma);
		shaderOutput.setBool("uncharted_tonemap", uncharted_tonemap);
	//}
	double gBufferTime, ssaoTime, deferredTime, forwardTime, postProcessTime, reconstructionTime, outputTime;
	int report = 0;
	Timer t, totalTime;
	// if shit doesn't show up try glm::mat4 model = glm::mat4(1.0f)

	while (!glfwWindowShouldClose(window))
	{
		totalTime.start();
		const double currentFrame = glfwGetTime();
		const float currentFrameFloat = (float)currentFrame;
		deltaTime = currentFrame - lastFrame;
		const float deltaTimeFloat = (float)deltaTime;
		lastFrame = currentFrame;
		const float lastFrameFloat = (float)lastFrame;

		process_input(window);

		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// gBuffer
		t.start();
		glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = camera.GetViewMatrix();
		glm::mat4 model = glm::mat4(1.0f);
		shaderGeometryPass.use();
		shaderGeometryPass.setMat4("projection", projection);
		shaderGeometryPass.setMat4("view", view);
		for (unsigned int i = 0; i < objectPositions.size(); i++)
		{
			model = glm::mat4(1.0f);
			model = glm::translate(model, objectPositions[i]);
			model = glm::scale(model, glm::vec3(0.25f));
			shaderGeometryPass.setMat4("model", model);
			nanosuit.Draw(shaderGeometryPass);
		}
		glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, forwardFBO);
		glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		t.stop();
		gBufferTime = t.get_time();
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// generate SSAO texture
		t.start();
		glBindFramebuffer(GL_FRAMEBUFFER, ssaoFBO);
		glClear(GL_COLOR_BUFFER_BIT);
		shaderSSAO.use();
		// Send kernel + rotation 
		for (unsigned int i = 0; i < 64; ++i)
			shaderSSAO.setVec3("samples[" + std::to_string(i) + "]", ssaoKernel[i]);
		shaderSSAO.setMat4("projection", projection);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gPosition);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, gNormal);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, noiseTexture);
		renderQuad();
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);


		// blur SSAO texture to remove noise
		glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurFBO);
		glClear(GL_COLOR_BUFFER_BIT);
		shaderSSAOBlur.use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, ssaoBuffer);
		renderQuad();
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);
		t.stop();
		ssaoTime = t.get_time();
		t.start();
		glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
		// lighting pass
		shaderLightingPass.use();

		// update lighting positions
		for (int i = 0; i < NR_LIGHTS - 2; ++i) {
			lightPositions[i].x += lightColors[i].x * i * cosf(currentFrameFloat) * 0.001f - tanf(lastFrameFloat * lightColors[i].z * i) * 0.0002f;
			lightPositions[i].y += lightColors[i].y * i * sinf(currentFrameFloat) * 0.0003f - sinf(deltaTimeFloat * lightColors[i].x * i) * 0.0001f;
			lightPositions[i].z += lightColors[i].z * i * sinf(currentFrameFloat) * 0.001f - cosf(currentFrameFloat * lightColors[i].y * i) * 0.0003f;
		}
		lightPositions[NR_LIGHTS - 2].x += lightColors[NR_LIGHTS - 2].x * (NR_LIGHTS - 2) * cosf(currentFrameFloat) * 0.01f - sinf(currentFrameFloat) * 0.03f;
		lightPositions[NR_LIGHTS - 1].z += lightColors[NR_LIGHTS - 1].z * (NR_LIGHTS - 1) * sinf(currentFrameFloat) * 0.01f;


		// send light relevant uniforms
		for (unsigned int i = 0; i < lightPositions.size(); i++)
		{
			shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].Position", glm::vec3(camera.GetViewMatrix() * glm::vec4(lightPositions[i], 1.0)));
			shaderLightingPass.setVec3("lights[" + std::to_string(i) + "].Color", lightColors[i]);
			// update attenuation parameters and calculate radius
			shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].Linear", linear);
			shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].Quadratic", quadratic);
			// then calculate radius of light volume/sphere
			const float maxBrightness = std::fmaxf(std::fmaxf(lightColors[i].r, lightColors[i].g), lightColors[i].b);
			float radius = (-linear + std::sqrt(linear * linear - 4 * quadratic * (constant - (256.0f / 5.0f) * maxBrightness))) / (2.0f * quadratic);
			shaderLightingPass.setFloat("lights[" + std::to_string(i) + "].Radius", radius);
		}
		shaderLightingPass.setVec3("viewPos", camera.Position);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gPosition);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, gNormal);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_2D, ssaoBufferBlur);
		renderQuad();
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);
		t.stop();
		deferredTime = t.get_time();
		t.start();
		glBindFramebuffer(GL_FRAMEBUFFER, forwardFBO);
		glClear(GL_COLOR_BUFFER_BIT);
		// render lights on top of scene
		shaderForward.use();
		shaderForward.setMat4("projection", projection);
		shaderForward.setMat4("view", view);
		for (unsigned int i = 0; i < lightPositions.size(); ++i)
		{
			
			model = glm::mat4(1.0f);
			model = glm::translate(model, lightPositions[i]);
			model = glm::scale(model, point_lights_size);
			shaderForward.setMat4("model", model);
			shaderForward.setVec3("lightColor", lightColors[i]);
			renderCube();
		}
		t.stop();
		forwardTime = t.get_time();
		t.start();

		if (laplace_pipeline) {
			// post processing
			// ---------------
			glBindFramebuffer(GL_FRAMEBUFFER, laplacePostProcessFBO);
			glDisable(GL_DEPTH_TEST);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
			shaderLaplacePostProcess.use();
			glClear(GL_COLOR_BUFFER_BIT);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, deferredOutput);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, forwardOutput);
			shaderLaplacePostProcess.setBool("edges", laplace_edges);
			shaderLaplacePostProcess.setBool("transparency", laplace_transparency);
			renderQuad();

			// reconstruct gradients
			// ---------------------
			// input none, but reads pixels from bound framebuffer
			// output laplaceReconstructionOutput
			reconstructor->reconstruct_from_gradients(laplaceReconstructionOutput);


			// output
			// ------
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glDisable(GL_DEPTH_TEST);
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			shaderLaplaceReconstructionOutput.use();
			shaderLaplaceReconstructionOutput.setBool("uncharted_tonemap", uncharted_tonemap);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, laplaceReconstructionOutput);
			renderQuad();
			glEnable(GL_DEPTH_TEST);

		}
		else {
			// post processing
			// ---------------
			glBindFramebuffer(GL_FRAMEBUFFER, postProcessFBO);
			//glDisable(GL_DEPTH_TEST);
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			shaderPostProcess.use();
			glClear(GL_COLOR_BUFFER_BIT);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, deferredOutput);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, forwardOutput);
			shaderPostProcess.setBool("edges", post_process_edges);
			shaderPostProcess.setBool("transparency", post_process_transparency);
			renderQuad();

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glDisable(GL_DEPTH_TEST);
			glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			shaderOutput.use();
			shaderOutput.setBool("uncharted_tonemap", uncharted_tonemap);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, postProcessOutput);
			renderQuad();
			glEnable(GL_DEPTH_TEST);
		}

		
		t.stop();
		postProcessTime = t.get_time();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		// ---------------

		glfwSwapBuffers(window);
		glfwPollEvents();
		totalTime.stop();
		if (print_timing) {
			if (report == 10) {
				report = 0;
				std::cout << "gBuffer: " << gBufferTime * 1000.0f << "\n";
				std::cout << "ssao: " << ssaoTime * 1000.0f << "\n";
				std::cout << "deferred: " << deferredTime * 1000.0f << "\n";
				std::cout << "forward: " << forwardTime * 1000.0f << "\n";
				std::cout << "post processing: " << postProcessTime * 1000.0f << "\n";
				std::cout << "total frame: " << totalTime.get_time() * 1000.0f << "\n\n";
			}
			++report;
		}
	}
	delete reconstructor;

	glfwTerminate();
	return EXIT_SUCCESS;
}

void renderCube()
{
	// initialize (if necessary)
	if (cubeVAO == 0)
	{
		float vertices[] = {
			// back face
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
			1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
			-1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
			-1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
			// front face
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			// left face
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};
		glGenVertexArrays(1, &cubeVAO);
		glGenBuffers(1, &cubeVBO);
		// fill buffer
		glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		// link vertex attributes
		glBindVertexArray(cubeVAO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	// render Cube
	glBindVertexArray(cubeVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);
}


void renderQuad()
{
	if (quadVAO == 0)
	{
		float quadVertices[] = {
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}

void process_input(GLFWwindow* window) {
	// if escape was pressed, suggest closing the window
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.ProcessKeyboard(FORWARD, deltaTime + 1.0f);
		else
			camera.ProcessKeyboard(FORWARD, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.ProcessKeyboard(BACKWARD, deltaTime + 1.0f);
		else
			camera.ProcessKeyboard(BACKWARD, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.ProcessKeyboard(LEFT, deltaTime + 1.0f);
		else
			camera.ProcessKeyboard(LEFT, deltaTime);
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.ProcessKeyboard(RIGHT, deltaTime + 1.0f);
		else
			camera.ProcessKeyboard(RIGHT, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
	{
		uncharted_tonemap = !uncharted_tonemap;
		std::cout << "uncharted tonemapping equals " << uncharted_tonemap << "\n\n";
	}

	if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
	{
		laplace_pipeline = !laplace_pipeline;
		if (laplace_pipeline)	std::cout << "using gradient reconstruction postprocessing pipeline\n\n";
		else std::cout << "using postprocessing pipeline\n\n";
	}

	if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
	{
		print_timing = !print_timing;
		std::cout << "print timing equals " << print_timing << "\n\n";
	}
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	// set window to specified dimensions
	glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	const float _xpos = (float)xpos;
	const float _ypos = (float)ypos;
	if (firstMouse) {
		lastX = _xpos;
		lastY = _ypos;
		firstMouse = false;
	}

	const float xoffset = _xpos - lastX;
	const float yoffset = lastY - _ypos;
	lastX = _xpos;
	lastY = _ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);

}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.ProcessMouseScroll((float)yoffset);
}

unsigned int loadTexture(char const * path)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);

	int width, height, nrComponents;
	unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
	if (data)
	{
		GLenum format;
		if (nrComponents == 1) format = GL_RED;
		else if (nrComponents == 3)	format = GL_RGB;
		else if (nrComponents == 4)	format = GL_RGBA;
		else format = GL_RGB;

		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		stbi_image_free(data);
	}
	else
	{
		std::cout << "Texture failed to load at path: " << path << std::endl;
		stbi_image_free(data);
	}

	return textureID;
}