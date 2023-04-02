#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxAudioAnalyzer.h"

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

		struct MelBand{
			glm::vec4 value;
		};

		int sampleRate;
		int bufferSize;
		int channels;

		ofSoundStream soundStream;
		ofxAudioAnalyzer audioAnalyzer;

		ofBufferObject melBandsBuffer;

		ofShader compute_audio;
		ofTexture audio_texture;

		float smoothing;
};
