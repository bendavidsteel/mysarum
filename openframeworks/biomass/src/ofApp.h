#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxVideoRecorder.h"
#include "ofxVolumetrics.h"
#include "ofxOpenCv.h"

enum Spawn{
	RANDOM,
	CIRCLE,
	RING
};

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void exit();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
		void audioIn(float * input, int bufferSize, int nChannels);

		ofxVideoRecorder vidRecorder;
		ofSoundStream soundStream;
		bool bRecording;
		int sampleRate;
		int channels;
		string fileName;
		string fileExt;

		void recordingComplete(ofxVideoRecorderOutputFileCompleteEventArgs& args);

		// both
		ofShader compute_flow;
		ofShader renderer;

		ofxTexture3d flowMap;

		// slime
		struct Agent{
			glm::vec4 pos;
			glm::vec4 vel;
			glm::vec4 attributes;
		};

		struct Species{
			glm::vec4 colour;
			glm::vec4 sensorAttributes;
			glm::vec4 movementAttributes;
		};

		ofShader compute_agents;
		ofShader compute_decay;
		ofShader compute_diffuse;
		
		ofPixels pixels;
		ofFbo fbo;

		vector<Agent> particles;
		ofBufferObject particlesBuffer;
		vector<Species> allSpecies;
		ofBufferObject allSpeciesBuffer;
		ofxTexture3d trailMap;

		int volWidth;
		int volHeight;
		int volDepth;

		float diffuseRate;
		float decayRate;
		float trailWeight;

		// reaction diffusion
		ofShader compute_diffusion;
		ofShader compute_feedkill;
		ofShader compute_reaction;

		ofTexture diffusionMap;
		ofTexture feedkillMap;
		ofTexture reactionMap;

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
		ofTexture opticalFlowTexture;
		
		int blurAmount;
		float cvDownScale;
		bool bContrastStretch;
		float minLengthSquared;
};
