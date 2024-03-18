#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxGui.h"

class Boids{

	public:
		void setup(int numBins);
		void update();
		void draw(int _x, int _y, int _w, int _h);
		void drawGui();

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
		ofEasyCam camera;
		ofVbo vbo;
		ofFbo fbo;

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
