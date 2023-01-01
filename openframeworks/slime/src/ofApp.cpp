#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
	img.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);	

	slime.setup(numAgents);
}

//--------------------------------------------------------------
void ofApp::update(){
	slime.update();
}

//--------------------------------------------------------------
void ofApp::draw(){
	ofPixels & pixels = img.getPixels();
	slime.draw(pixels);
	img.update();
	img.draw(0, 0, ofGetWidth(), ofGetHeight());
}
