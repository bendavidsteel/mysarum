#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){
	smoothing = 0.;

	sampleRate = 44100;
    bufferSize = 512;
    int channels = 1;
    
    audioAnalyzer.setup(sampleRate, bufferSize, channels);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA16);

	renderer.load("renderer.vert", "renderer.frag");

}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	dissonance = audioAnalyzer.getValue(DISSONANCE, 0, smoothing);
    
    spectrum = audioAnalyzer.getValues(SPECTRUM, 0, smoothing);
    melBands = audioAnalyzer.getValues(MEL_BANDS, 0, smoothing);
    mfcc = audioAnalyzer.getValues(MFCC, 0, smoothing);
    hpcp = audioAnalyzer.getValues(HPCP, 0, smoothing);
    
    tristimulus = audioAnalyzer.getValues(TRISTIMULUS, 0, smoothing);

	//-:ANALYZE SOUNDBUFFER:
    audioAnalyzer.analyze(soundBuffer);

	int workGroupSize = 32;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	// modulate using a v simple neural net with weights from the audio analysis

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
	audioAnalyzer.exit();
    player.stop();
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