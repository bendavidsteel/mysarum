#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){
	float_strength = 1;

	reactionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	reactionMap.bindAsImage(0, GL_READ_WRITE);

	trailMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	trailMap.bindAsImage(1, GL_READ_WRITE);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA16);

	compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();

	renderer.load("renderer.vert", "renderer.frag");

}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();
	int workGroupSize = 32;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.setUniform1f("float_strength", float_strength);
	compute_reaction.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_reaction.end();
}

//--------------------------------------------------------------
void ofApp::draw() {
	// flowMap.draw(0, 0, ofGetWidth(), ofGetHeight());

	fbo.begin();
	ofClear(255,255,255, 0);
	renderer.begin();
	renderer.setUniform3f("colourA", 1., 0., 0.);
	renderer.setUniform3f("colourB", 0., 1., 0.);
	renderer.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	renderer.setUniform3f("light", ofGetWidth(), 0., 5.);
	renderer.setUniform1f("chem_height", 1.);
	renderer.setUniform1f("trail_height", 2.);
	ofSetColor(255);
	ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	renderer.end();
	fbo.end();
	fbo.draw(0, 0, ofGetWidth(), ofGetHeight());

	ofDrawBitmapString(ofGetFrameRate(),20,20);
}

void ofApp::exit(){

	
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}