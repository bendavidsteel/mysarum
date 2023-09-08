#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxVideoRecorder.h"
#include "ofxOpenCv.h"
#include "ofxAudioAnalyzer.h"
#include "ofxBPMDetector.h"
#include "ofxMidi.h"
#include "ofxNetwork.h"

#include "biomass.h"
#include "evolution.h"

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
		void audioIn(ofSoundBuffer & input);
		void newMidiMessage(ofxMidiMessage& eventArgs);
		void newInput(int key);

		Biomass biomass;
		Evolution evolution;

		ofxVideoRecorder vidRecorder;
		ofSoundStream soundStream;
		ofxMidiIn midiIn;

		ofImage maskImage;

		ofxUDPManager udpConnection;

		int sampleRate;
		int bufferSize;
		int channels;
		float volume;
		AudioType audioType;

		// both
		ofShader postprocess;
		ofShader preprocess;

		ofPixels pixels;
		ofFbo fbo;

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
		ofxAudioAnalyzer audioAnalyzer;
		ofxBPMDetector bpmDetector;

		int audioArraySize;
		vector<Component> audioArray;

		float lowSmoothing;
		float highSmoothing;

		// variables
		float time_of_day;
		float time_of_month;

		bool bReloadShader;
};
