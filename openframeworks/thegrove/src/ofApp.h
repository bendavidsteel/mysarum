#pragma once

#include "ofMain.h"
#include "ofxMidi.h"
#include "ofxPDSP.h"
#include "ofxGui.h"
// #include "synth.h"
#include "boids.h"
#include "physarum3d.h"
#include "selforganising.h"
#include "noise.h"
#include "synths.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
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
		
		Physarum physarum;
		Boids boids;
		SelfOrganising selfOrganising;
		Noise noise;

		int width, height, depth;
		float windStrength;
		ofVec2f windDirection;

		ofCamera cam;

        int numBins;

        pdsp::Engine engine;

		vector<BoidsSynth> boidSynths;
		vector<SelfOrganisingSynth> selfOrganisingSynths;
		PhysarumSynth physarumSynth;
		
        ofxPanel                    gui;
};
