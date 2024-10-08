#include "volumetrics.h"

using namespace glm;

Volumetrics::~Volumetrics() {
	destroy();
}

Volumetrics::Volumetrics() {
	quality = vec3(1.0);
	threshold = 1.0 / 255.0;
	density = 1.0;
	volWidth = renderWidth = 0;
	volHeight = renderHeight = 0;
	volDepth = 0;
	bIsInitialized = false;
	bDrawDebugVolume = false;

	vector<vec3> vertices {
		// front side
		vec3(1.0, 1.0, 1.0),
		vec3(0.0, 1.0, 1.0),
		vec3(1.0, 0.0, 1.0),
		vec3(0.0, 0.0, 1.0),
		// right side
		vec3(1.0, 1.0, 0.0),
		vec3(1.0, 1.0, 1.0),
		vec3(1.0, 0.0, 0.0),
		vec3(1.0, 0.0, 1.0),
		// top side
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, 1.0, 1.0),
		vec3(1.0, 1.0, 0.0),
		vec3(1.0, 1.0, 1.0),		
		// left side
		vec3(0.0, 1.0, 1.0),
		vec3(0.0, 1.0, 0.0),		
		vec3(0.0, 0.0, 1.0),
		vec3(0.0, 0.0, 0.0),
		// bottom side
		vec3(1.0, 0.0, 1.0),
		vec3(0.0, 0.0, 1.0),
		vec3(1.0, 0.0, 0.0),
		vec3(0.0, 0.0, 0.0),		
		// back side
		vec3(0.0, 1.0, 0.0),
		vec3(1.0, 1.0, 0.0),	
		vec3(0.0, 0.0, 0.0),
		vec3(1.0, 0.0, 0.0)		
	};	
	vector<unsigned int> indices {
		// front side
		0, 2, 1, 1, 2, 3,
		// right side
		4, 6, 5, 5, 6, 7,
		// top side
		8, 10, 9, 9, 10, 11,
		// left side
		12, 14, 13, 13, 14, 15,
		// bottom side
		16, 18, 17, 17, 18, 19,
		// back side
		20, 22, 21, 21, 22, 23
	};
	volumeMesh.addVertices(vertices);
	volumeMesh.addIndices(indices);
}

void Volumetrics::setup(int w, int h, int d, vec3 voxelSize, bool usePowerOfTwoTexSize) {
	volumeShader.unload();
	volumeShader.load("raycast.vert", "raycast.frag");

	bIsPowerOfTwo = usePowerOfTwoTexSize;

	volWidthPOT = volWidth = renderWidth = w;
	volHeightPOT = volHeight = renderHeight = h;
	volDepthPOT = volDepth = d;

	if (bIsPowerOfTwo) {
		volWidthPOT = ofNextPow2(w);
		volHeightPOT = ofNextPow2(h);
		volDepthPOT = ofNextPow2(d);

		ofLogVerbose() << "Volumetrics::setup(): Using power of two texture size. Requested: " << w << "x" << h << "x" << d << ". Actual: " << volWidthPOT << "x" << volHeightPOT << "x" << volDepthPOT << ".\n";
	}

	fboRender.allocate(w, h, GL_RGBA8);
	voxelRatio = voxelSize;

	bIsInitialized = true;
}

void Volumetrics::destroy() {
	volumeShader.unload();

	volWidth = renderWidth = 0;
	volHeight = renderHeight = 0;
	volDepth = 0;
	bIsInitialized = false;
}

void Volumetrics::drawVolume(float x, float y, float z, float size, int zTexOffset) {
	vec3 volumeSize = voxelRatio * vec3(volWidth, volHeight, volDepth);
	float maxDim = glm::max(glm::max(volumeSize.x, volumeSize.y), volumeSize.z);
	volumeSize = volumeSize * size / maxDim;

	drawVolume(x, y, z, volumeSize.x, volumeSize.y, volumeSize.z, zTexOffset);
}

void Volumetrics::drawVolume(float x, float y, float z, float w, float h, float d, int zTexOffset) {
	updateRenderDimentions();

	// store current color
	GLint color[4];
	glGetIntegerv(GL_CURRENT_COLOR, color);

	// store current cull mode
	GLint cull_mode;
	glGetIntegerv(GL_FRONT_FACE, &cull_mode);

	// set fbo cull mode
	mat4 matModelview = ofGetCurrentMatrix(OF_MATRIX_MODELVIEW);
	ofVec3f scale, t; ofQuaternion a, b;
	ofMatrix4x4(matModelview).decompose(t, a, scale, b);
	GLint cull_mode_fbo = (scale.x * scale.y * scale.z) > 0 ? GL_CCW : GL_CW;

	// raycasting pass
	fboRender.begin(OF_FBOMODE_NODEFAULTS);
	{
		// fix flipped y-axis
		ofSetMatrixMode(OF_MATRIX_PROJECTION);
		ofScale(1, -1, 1);
		ofSetMatrixMode(OF_MATRIX_MODELVIEW);
		ofScale(1, -1, 1);

		ofClear(0, 0);

		vec3 cubeSize(w, h, d);
		vec3 cubePos(x, y, z);
		ofTranslate(cubePos - cubeSize / 2.f);
		ofScale(cubeSize.x, cubeSize.y, cubeSize.z);

		volumeShader.begin();
		volumeShader.setUniform3f("vol_d", vec3(volWidth, volHeight, volDepth)); //dimensions of the volume texture
		volumeShader.setUniform3f("vol_d_pot", vec3(volWidthPOT, volHeightPOT, volDepthPOT)); //dimensions of the volume texture power of two
		volumeShader.setUniform2f("bg_d", vec2(renderWidth, renderHeight)); // dimensions of the background texture
		volumeShader.setUniform1f("zoffset", zTexOffset); // used for animation so that we dont have to upload the entire volume every time
		volumeShader.setUniform1f("quality", quality.z); // 0..1
		volumeShader.setUniform1f("density", density); // 0..1
		volumeShader.setUniform1f("threshold", threshold);
		if (ofIsGLProgrammableRenderer()) {
			volumeShader.setUniformMatrix4f("modelViewMatrixInverse", inverse(ofGetCurrentMatrix(OF_MATRIX_MODELVIEW)));
		}

		glFrontFace(cull_mode_fbo);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_FRONT);
		volumeMesh.drawFaces();
		glDisable(GL_CULL_FACE);
		glFrontFace(cull_mode);

		volumeShader.end();

		if (bDrawDebugVolume) {
			glFrontFace(cull_mode_fbo);
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
			volumeMesh.drawWireframe();
			glDisable(GL_CULL_FACE);
			glFrontFace(cull_mode);
		}		
	}
	fboRender.end();

	ofPushView();

	glColor4iv(color);
	ofSetupScreenOrtho();		
	fboRender.draw(0, 0, ofGetWidth(), ofGetHeight());

	ofPopView();
}

void Volumetrics::updateRenderDimentions() {
	if ((int)(ofGetWidth() * quality.x) != renderWidth) {
		renderWidth = ofGetWidth() * quality.x;
		renderHeight = ofGetHeight() * quality.x;
		fboRender.allocate(renderWidth, renderHeight, GL_RGBA);
	}
}

void Volumetrics::setXyQuality(float q) {
	quality.x = MAX(q, 0.01);

	updateRenderDimentions();
}
void Volumetrics::setZQuality(float q) {
	quality.z = MAX(q, 0.01);
}
void Volumetrics::setThreshold(float t) {
	threshold = ofClamp(t, 0.0, 1.0);
}
void Volumetrics::setDensity(float d) {
	density = MAX(d, 0.0);
}
void Volumetrics::setRenderSettings(float xyQuality, float zQuality, float dens, float thresh) {
	setXyQuality(xyQuality);
	setZQuality(zQuality);
	setDensity(dens);
	setThreshold(thresh);
}

void Volumetrics::setDrawDebugVolume(bool b) {
	bDrawDebugVolume = b;
}

bool Volumetrics::isInitialized() {
	return bIsInitialized;
}
int Volumetrics::getVolumeWidth() {
	return volWidth;
}
int Volumetrics::getVolumeHeight() {
	return volHeight;
}
int Volumetrics::getVolumeDepth() {
	return volDepth;
}
int Volumetrics::getRenderWidth() {
	return renderWidth;
}
int Volumetrics::getRenderHeight() {
	return renderHeight;
}
float Volumetrics::getXyQuality() {
	return quality.x;
}
float Volumetrics::getZQuality() {
	return quality.z;
}
float Volumetrics::getThreshold() {
	return threshold;
}
float Volumetrics::getDensity() {
	return density;
}
const ofFbo& Volumetrics::getFbo() const {
	return fboRender;
}
