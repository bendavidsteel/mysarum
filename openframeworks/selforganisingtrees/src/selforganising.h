#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxSpatialHash.h"
// #include "environment.h"
#include "shadowpropagation.h"
// #include "spacecolonization.h"
#include "trees.h"

struct AcropetalState {
    shared_ptr<Metamer> metamer;
    float v;
    int treeIdx;
};

struct BasipetalState {
    shared_ptr<Metamer> metamer;
    int treeIdx;
    bool processedChildren;
};


class SelfOrganising {

	public:
		SelfOrganising();
		void setup();
		void update();
		void draw();

		Tree addTree(ofVec3f pos, ofVec3f dir, int idx);
		void startAcropetalPass();
		void acropetalPass();
		void startBasipetalPass();
		void basipetalPass();
		void addNewTerminalShoot(shared_ptr<Metamer> metamer, float v, int treeIdx);
		void addNewAxillaryShoot(shared_ptr<Metamer> metamer, float v, int treeIdx);
		void addNewShoot(shared_ptr<Metamer> metamer, float v, ofVec3f defaultDir, ofVec3f growthDir, bool isTerminal, int treeIdx);
        shared_ptr<Metamer> addMetamer(shared_ptr<Metamer> parent_metamer, ofVec3f pos, ofVec3f dir, bool isTerminal, int treeIdx);

		ShadowPropagation environment;

		vector<ofVec3f> metamerPositions;
		vector<glm::vec3> metamerVertices;
		vector<unsigned int> metamerIndices;
		vector<ofFloatColor> metamerColors;
        vector<float> metamerWidths;
		int metamerIdx;

		std::stack<BasipetalState> basipetalStack;
		std::stack<AcropetalState> acropetalStack;

		vector<Tree> trees;

		ofVbo metamerVbo;

        ofShader shader;
		ofEasyCam cam;
		float mapSize;

		int processStep;
		int basipetalCount, maxBasipetalCount;
		int acropetalCount, maxAcropetalCount;
};
