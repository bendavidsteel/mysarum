#pragma once

#include "ofMain.h"
#include "ofxDlib.h"
#include "ofxGraph.h"

class ofApp : public ofBaseApp{
	public:
	
		void setup();
		void update();
		void draw();

        ofVideoGrabber 		vidGrabber;
		
		ofTexture topTexture;
		ofTexture bottomTexture;

		vector<float> topMeanBuffer;
		vector<float> bottomMeanBuffer;
		vector<float> timeBuffer;

		vector<float> topHeartbeatBuffer;
		vector<float> bottomHeartbeatBuffer;

		vector<float> topPrunedFFT;
		vector<float> topPrunedFreqs;
		vector<float> bottomPrunedFFT;
		vector<float> bottomPrunedFreqs;

		float topBPM;
		float bottomBPM;

		ofxGraph rawTopGraph;
		ofxGraph rawBottomGraph;
		ofxGraph topHeartBeatGraph;
		ofxGraph bottomHeartBeatGraph;

		int maxBufferSize;
		bool bNewFrame;
};

void getHeartBeat(vector<float> buffer, vector<float> timeBuffer, float & bpm, float & heartbeat, vector<float> & pruned_fft, vector<float> & pruned_freqs);