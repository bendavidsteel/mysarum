#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxVideoRecorder.h"

enum InitialPattern{
	CIRCLE,
	RANDOM
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

		ofShader compute_flow;
		ofShader compute_feedkill;
		ofShader compute_reaction;
		ofShader renderer;

		ofPixels pixels;
		ofFbo fbo;

		ofTexture flowMap;
		ofTexture feedkillMap;
		ofTexture reactionMap;

		int flowSizeFactor;
};
