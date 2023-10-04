#include "ofApp.h"
#include "ofConstants.h"

ofApp::ofApp():
	nodeHash(nodePositions),
	attractorHash(attractors)
{
}

//--------------------------------------------------------------
void ofApp::setup(){
	ofSetFrameRate(60);

	// randomly create attractors
	int numAttractors = 10000;
	for (int i = 0; i < numAttractors; i++) {
		attractors.push_back(ofVec2f(ofRandomWidth(), ofRandomHeight()));
	}

	nodes.setMode(OF_PRIMITIVE_LINES);

	// create initial nodes
	int numNodes = int(ofRandom(3, 8));
	nodePositions.resize(numNodes);
	nodeAttractions.resize(numNodes);
	for (int i = 0; i < numNodes; i++) {
		nodePositions[i] = ofVec2f(ofRandomWidth(), ofRandomHeight());
		nodeAttractions[i] = ofVec2f(0, 0);

		nodes.addVertex(ofPoint(nodePositions[i].x, nodePositions[i].y, 0.));
		nodes.addVertex(ofPoint(nodePositions[i].x, nodePositions[i].y, 0.));
		nodes.addIndex(2*i);
		nodes.addIndex(2*i+1);
	}

	attractionDistance = ofGetWidth() / 4;
	killDistance = ofGetWidth() / 1000;
	segmentLength = ofGetWidth() / 1000;
}

//--------------------------------------------------------------
void ofApp::update(){
	// construct node spatial hash tree
	nodeHash.buildIndex();
	ofx::KDTree<ofVec2f>::SearchResults nodeSearchResults;

	// loop through attractors and find nearest node, and add attractor force
	for (int i = 0; i < attractors.size(); i++) {
		nodeSearchResults.clear();
		nodeSearchResults.resize(1);

		nodeHash.findNClosestPoints(attractors[i], 1, nodeSearchResults);
		int closestNode = nodeSearchResults[0].first;
		nodeAttractions[closestNode] += (attractors[i] - nodePositions[closestNode]).normalize();
	}

	// create attractor spatial hash
	attractorHash.buildIndex();
	ofx::KDTree<ofVec2f>::SearchResults attractorSearchResults;

	// loop through nodes with attraction, and add a node in the direction of the attraction
	int numNodes = nodePositions.size();
	for (int i = 0; i < numNodes; i++) {
		if (nodeAttractions[i].length() > 0) {
			ofVec2f newPos = nodePositions[i] + nodeAttractions[i].normalize() * segmentLength;

			nodes.addVertex(ofPoint(nodePositions[i].x, nodePositions[i].y, 0.));
			nodes.addVertex(ofPoint(newPos.x, newPos.y, 0.));
			nodes.addIndex(nodes.getNumVertices() - 2);
			nodes.addIndex(nodes.getNumVertices() - 1);

			nodeAttractions.push_back(ofVec2f(0, 0));
			nodePositions.push_back(newPos);

			// check if new node is close to an attractor, and remove attractor if so
			attractorSearchResults.clear();
			attractorSearchResults.resize(1);

			attractorHash.findPointsWithinRadius(nodePositions.back(), killDistance, attractorSearchResults);
			if (attractorSearchResults.size() > 0) {
				attractors.erase(attractors.begin() + attractorSearchResults[0].first);
			}

			nodeAttractions[i] = ofVec2f(0, 0);
		}
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	// draw nodes
	ofSetColor(255);

	nodes.draw();

	// draw attractors
	// ofSetColor(255, 0, 0);
	// for (int i = 0; i < attractors.size(); i++) {
	// 	ofDrawCircle(attractors[i], 1);
	// }

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