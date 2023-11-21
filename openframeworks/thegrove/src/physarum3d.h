#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxVideoRecorder.h"
#include "ofxVolumetrics.h"
#include "volumetrics.h"
#include "ofxGui.h"

enum AgentSpawn{
	RANDOM,
	CENTRE,
	RING,
	EDGES
};

class Physarum{

	public:
		void setup(int _numBins, int _width, int _height, int _depth);
		void update(float windStrength, ofVec2f windDirection, float activity);
		void draw();
		void drawGui(int x, int y);
		void exit();

		void renderUniformsChanged(ofAbstractParameter &e);
		void swapTrailMaps();

		vector<float> getMaxSenseHist();
		vector<float> getAvgSenseHist();
		vector<float> getTurnSpeedHist();

		struct Agent{
			glm::vec4 pos;
			glm::vec4 vel;
			glm::vec4 attributes;
			glm::vec4 state;
		};

		struct Species{
			glm::vec4 colour;
			glm::vec4 sensorAttributes;
			glm::vec4 movementAttributes;
		};

		ofShader compute_agents;
		ofShader compute_decay;
		ofShader compute_diffuse;
		ofShader compute_flow;
		ofShader renderer;

		ofPixels pixels;
		ofFbo fbo;

		vector<Agent> particles;
		ofBufferObject particlesBuffer;
		ofBufferObject particlesBuffer2;
		vector<Species> allSpecies;
		ofBufferObject allSpeciesBuffer;
		ofxTexture3d trailMap;
		ofxTexture3d trailMap2;
		ofxTexture3d flowMap;

		bool useFirstTexture;

		Volumetrics volume;

		int volWidth, volHeight, volDepth;
		int width, height, depth;

		ofParameter<float> diffuseRate;
		ofParameter<float> decayRate;
		ofParameter<float> trailWeight;

		ofParameter<float> xyQuality;
		ofParameter<float> zQuality;
		ofParameter<float> density;
		ofParameter<float> threshold;

		ofParameter<float> turnSpeed;
		ofParameter<float> moveSpeed;
		ofParameter<float> sensorAngleRad;
		ofParameter<float> sensorOffsetDist;

		ofParameterGroup renderUniforms;
		ofParameterGroup trailUniforms;
		ofParameterGroup speciesUniforms;

		ofxPanel gui;

		vector<float> maxSenseHist;
		vector<float> avgSenseHist;
		vector<float> turnSpeedHist;
		int numBins;
};
