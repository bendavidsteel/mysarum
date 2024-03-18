#pragma once

#include "ofMain.h"
#include "ofxGui.h"

#include "ColorWheelScheme.h"

#include "arpeggio.h"
#include "synths.h"

using namespace ofxColorTheory;
using namespace std;

class Hypercycles : public pdsp::Patchable{
	public:
		
	Hypercycles() { patch(); } // default constructor
	Hypercycles( const Hypercycles & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
	void patch();
	pdsp::Patchable & out_signal();
	void setup(int width, int height);
	void update();
	void draw();
	void drawGui();

	ofShader shader;
	ofPlanePrimitive plane;
	ofImage img;
	ofxPanel gui;

	int width, height;

	vector<int> cells;

	int numSpecies;
	vector<int> speciesFriends;
	vector<ofColor> speciesColours;
	shared_ptr<ColorWheelScheme> scheme;

	ofParameter<float> birthRate;
	ofParameter<float> deathRate;
	ofParameter<float> moveRate;
	ofParameter<float> replicateRate;
	ofParameter<float> catalyticSupport;
	ofParameter<float> synthAddRate;

	vector<Arpeggio> arpeggios;

	pdsp::ParameterGain gain;
};
