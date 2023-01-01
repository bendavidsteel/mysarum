#pragma once

#include "ofMain.h"
#include "slime.h"

const int numAgents = 1;

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		Slime slime;
		ofImage img;
};
