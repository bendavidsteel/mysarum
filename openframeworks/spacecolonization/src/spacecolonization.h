#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxSpatialHash.h"

struct Node {
	vector<shared_ptr<Node>> children;
	shared_ptr<Node> parent;
	int idx;
	float width;
	float opacity;
};

class SpaceColonization {

	public:
		SpaceColonization();
		void setup();
		void update();
		void draw();

        void addNode(ofVec3f posA, ofVec3f posB, int idx, shared_ptr<Node> parent);
        void addNode(ofVec3f posA, ofVec3f posB, int idx);
        void addNodeData(ofVec3f posA, ofVec3f posB, int idx);
        void recurWidth(shared_ptr<Node> node);

		vector<ofVec3f> nodePositions;
		vector<glm::vec3> nodeVertices;
		vector<unsigned int> nodeIndices;
		vector<ofFloatColor> nodeColors;
        vector<float> nodeWidths;
		vector<ofVec3f> nodeAttractions;
		int nodeIdx;

		vector<ofVec3f> attractors;

		ofVbo nodeVbo;
		vector<shared_ptr<Node>> nodeRoots;
		vector<shared_ptr<Node>> nodeTree;

		float attractionDistance;
		float killDistance;
		float segmentLength;
		float initialWidth;
		float maxWidth;
		float growthFactor;

		ofx::KDTree<ofVec3f> nodeHash;
		ofx::KDTree<ofVec3f> attractorHash;

        ofShader shader;
		ofCamera cam;
};
