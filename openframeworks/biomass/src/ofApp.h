#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxVideoRecorder.h"
#include "ofxOpenCv.h"
#include "ofxAudioAnalyzer.h"
#include "ofxMidi.h"
#include "ofxNetwork.h"

enum Spawn{
	RANDOM=0,
	CIRCLE=1,
	RING=2,
	SMALL_RING=3,
	VERTICAL_LINE=4,
	HORIZONTAL_LINE=5
};

enum AudioType{
	TIME=0,
	SPECTRAL=1,
	HEARTBEAT=2
};

class ofApp : public ofBaseApp, public ofxMidiListener{

	public:
		void setup();
		void update();
		void draw();
		void exit();

		void keyPressed(int key);
		void keyReleased(int key);
		void audioIn(ofSoundBuffer & input);
		void newMidiMessage(ofxMidiMessage& eventArgs);
		void newInput(int key);

		void copyVariables();
		void moveToVariables();
		void reSpawnAgents();
		void reSpawnReaction(bool keepPattern);

		ofxVideoRecorder vidRecorder;
		ofSoundStream soundStream;
		ofxMidiIn midiIn;

		ofxUDPManager udpConnection;

		int sampleRate;
		int bufferSize;
		int channels;
		float volume;
		AudioType audioType;

		// both
		ofShader compute_flow;
		ofShader renderer;
		ofShader simple_renderer;
		ofShader postprocess;

		ofFbo flowFbo;

		// slime
		struct Agent{
			glm::vec2 pos;
			glm::vec2 vel;
			glm::vec4 attributes;
		};

		struct Species{
			glm::vec4 colour;
			glm::vec4 movementAttributes;
			glm::vec4 sensorAttributes;
		};

		ofShader compute_agents;
		ofShader compute_decay;
		ofShader compute_diffuse;
		
		ofPixels pixels;
		ofFbo fbo;

		vector<Agent> particles;
		ofBufferObject particlesBuffer;
		vector<Species> allSpecies;
		vector<Species> newSpecies;
		ofBufferObject allSpeciesBuffer;
		ofTexture trailMap;

		float diffuseRate;
		float newDiffuseRate;
		float decayRate;
		float newDecayRate;
		float trailWeight;
		float newTrailWeight;

		// reaction diffusion
		ofShader compute_diffusion;
		ofShader compute_feedkill;
		ofShader compute_reaction;

		ofTexture diffusionMap;
		ofTexture reactionMap;
		ofFbo feedkillFbo;

		int flowSizeFactor;
		float floatStrength;

		// video and optical flow
		ofVideoGrabber 		vidGrabber;

        ofxCvColorImage			colorImg;
		ofxCvGrayscaleImage 	grayImage;
		
		ofxCvGrayscaleImage currentImage;
		cv::Mat previousMat;
		cv::Mat flowMat;

		ofPixels opticalFlowPixels;
		ofTexture optFlowTexture;
		
		int blurAmount;
		float cvDownScale;
		bool bContrastStretch;
		float minLengthSquared;

		// audio analysis
		struct Component{
			glm::vec4 value;
		};

		ofxAudioAnalyzer audioAnalyzer;

		int audioArraySize;
		vector<Component> audioArray;
		ofBufferObject audioBuffer;
		ofBufferObject pointsBuffer;

		ofShader compute_audio;
		ofTexture audioTexture;

		float lowSmoothing;
		float highSmoothing;

		vector<Component> points;
		vector<Component> newPoints;

		// variables
		float time_of_day;
		float time_of_month;

		float dayRate;
		float monthRate;
		float newDayRate;
		float chemHeight;
		float newChemHeight;
		float trailHeight;
		float newTrailHeight;

		float feedMin;
		float newFeedMin;
		float feedRange;
		float newFeedRange;

		float reactionFlowMag;
		float newReactionFlowMag;
		float agentFlowMag;
		float newAgentFlowMag;

		glm::vec3 colourA;
		glm::vec3 newColourA;
		glm::vec3 colourB;
		glm::vec3 newColourB;
		glm::vec3 colourC;
		glm::vec3 newColourC;
		glm::vec3 colourD;
		glm::vec3 newColourD;

		int display;

		bool bReSpawnAgents;
		bool bReSpawnReaction;
};
