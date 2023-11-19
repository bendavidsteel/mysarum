#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);
	selfOrganising.setup();
}

//--------------------------------------------------------------
void ofApp::update(){
	selfOrganising.update();
}

//--------------------------------------------------------------
void ofApp::draw() {
	selfOrganising.draw();

	// draw frame rate
	ofSetColor(255);
	ofDrawBitmapString(ofToString(ofGetFrameRate()), 10, 20);
}

void ofApp::exit(){
    
}



//--------------------------------------------------------------
void ofApp::audioIn(float *input, int bufferSize, int nChannels){
    
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