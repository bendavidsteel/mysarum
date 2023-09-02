#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxAudioAnalyzer.h"
#include "ofxNetwork.h"

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
		void audioIn(ofSoundBuffer & input);

		struct Component{
			glm::vec4 value;
		};

		int sampleRate;
		int bufferSize;
		int channels;

		ofSoundStream soundStream;
		ofxAudioAnalyzer audioAnalyzer;

		ofBufferObject audioBuffer;

		ofShader compute_audio;
		ofShader compute_flow;

		ofTexture audio_texture;
		ofTexture flowMap;

		float lowSmoothing;
		float highSmoothing;
		float numPoints;

		int audioBufferSize;
		vector<Component> rmsBuffer;

		ofxUDPManager udpConnection;
};
