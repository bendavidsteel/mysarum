#include "ofApp.h"
#include "ofConstants.h"

#define MAX_NODES 1000000

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

	shader.load("node.vert", "node.frag", "node.geom");

	// create node vbo
	nodeVertices.reserve(MAX_NODES * 2);
	nodeIndices.reserve(MAX_NODES * 2);
	nodeColors.reserve(MAX_NODES);
	nodeWidths.reserve(MAX_NODES);

	// create initial nodes
	int numInitialNodes = int(ofRandom(3, 8));
	nodePositions.resize(numInitialNodes);
	nodeAttractions.resize(numInitialNodes);
	for (int i = 0; i < numInitialNodes; i++) {
		ofVec2f pos = ofVec2f(ofRandomWidth(), ofRandomHeight());
		addNode(pos, pos, 2 * i, true);
	}

	nodeVbo.setVertexData(nodeVertices.data(), MAX_NODES * 2, GL_DYNAMIC_DRAW);
	nodeVbo.setIndexData(nodeIndices.data(), MAX_NODES * 2, GL_DYNAMIC_DRAW);
	nodeVbo.setColorData(nodeColors.data(), MAX_NODES, GL_DYNAMIC_DRAW);
	
	shader.begin();
	int widthAttLoc = shader.getAttributeLocation("nodeWidth");
	vbo.setAttributeData(widthAttLoc, nodeWidths, 1, MAX_NODES, GL_DYNAMIC_DRAW);
	shader.end();

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
	bool nodeAdded = false;
	int numNodes = nodePositions.size();
	for (int i = 0; i < numNodes; i++) {
		if (nodeAttractions[i].length() > 0) {
			nodeAdded = true;
			ofVec2f orgPos = nodePositions[i];
			ofVec2f newPos = nodePositions[i] + nodeAttractions[i].normalize() * segmentLength;

			addNode(orgPos, newPos, nodeIndices.back() + 1, false, );

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

	if (nodeAdded) {
		nodeVbo.updateVertexData(nodeVertices);
		nodeVbo.addIndices(nodeIndices);
		nodeVbo.addColors(nodeColors);
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	// draw nodes
	ofSetColor(255);

	nodeVbo.drawElements(GL_LINES, nodeVbo.getNumIndices());

	// draw attractors
	bool drawAttractors = false;
	if (drawAttractors)
	{
		ofSetColor(255, 0, 0);
		for (int i = 0; i < attractors.size(); i++) {
			ofDrawCircle(attractors[i], 1);
		}
	}

	// draw frame rate
	ofSetColor(255);
	ofDrawBitmapString(ofToString(ofGetFrameRate()), 10, 20);
}

void ofApp::exit(){
    
}

void ofApp::addNode(ofVec2f posA, ofVec2f posB, int idx, bool isRoot, Node* parent=NULL) {
	nodePositions[i] = posB;
	nodeAttractions[i] = ofVec2f(0, 0);

	nodeVertices.push_back(glm::vec3(posA.x, posA.y, 0.));
	nodeVertices.push_back(glm::vec3(posB.x, posB.y, 0.));
	nodeIndices.push_back(idx);
	nodeIndices.push_back(idx+1);
	nodeColors.push_back(ofFloatColor(1, 1, 1, 1));
	nodeColors.push_back(ofFloatColor(1, 1, 1, 1));

	Node node = Node();
	node.idx = i;
	node.width = 1;
	node.opacity = 1;

	if (isRoot) {
		nodeRoots.push_back(node);
	}
	else {
		parent->children.push_back(node);
	}
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