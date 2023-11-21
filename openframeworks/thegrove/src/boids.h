#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxGui.h"

class Boids{

	public:
		void setup(int _numBins, int _width, int _height, int _depth);
		void update(float windStrength, ofVec2f windDirection);
		void draw();
		void drawGui(int x, int y);

		vector<float> getAmpHistogram();
		vector<float> getAlignHistogram();
		vector<float> getPeerHistogram();
		
		struct Particle{
			glm::vec4 pos;
			glm::vec4 vel;
			glm::vec4 attr;
			ofFloatColor color;
		};

		ofShader compute;
		vector<Particle> particles;
		ofBufferObject particlesBuffer, particlesBuffer2;
		ofTexture histogramTexture;
		GLuint vaoID;
		ofVbo vbo;
		ofFbo fbo;

		int width, height, depth;

		ofxPanel gui;
		ofParameter<float> fov;
		ofParameter<float> attractionCoeff, alignmentCoeff, repulsionCoeff;
		ofParameter<float> attractionMaxDist, alignmentMaxDist, repulsionMaxDist;
		ofParameter<float> maxSpeed;
		ofParameter<float> randomForce;
		ofParameter<float> kuramotoStrength, kuramotoMaxDist;
		ofParameterGroup shaderUniforms;
		ofParameter<float> fps;
		ofParameter<bool> dirAsColor;

		int numParticles;
		int numBins;
		int numHists;
		vector<float> ampHist;
		vector<float> alignHist;
		vector<float> peerHist;
};
