#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxSpatialHash.h"

struct Node {
	vector<Node> children;
	int idx;
	float width;
	float opacity;
};

class ofApp : public ofBaseApp{

	public:
		ofApp();
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
		void audioIn(float * input, int bufferSize, int nChannels);

		vector<ofVec2f> nodePositions;
		vector<glm::vec3> nodeVertices;
		vector<unsigned int> nodeIndices;
		vector<ofFloatColor> nodeColors;
		vector<ofVec2f> nodeAttractions;

		vector<ofVec2f> attractors;

		ofVbo nodeVbo;
		vector<Node> nodeRoots;

		float attractionDistance;
		float killDistance;
		float segmentLength;

		ofx::KDTree<ofVec2f> nodeHash;
		ofx::KDTree<ofVec2f> attractorHash;
};
