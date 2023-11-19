#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
#include "ofxSpatialHash.h"
// #include "environment.h"
// #include "shadowpropagation.h"
#include "spacecolonization.h"
#include "trees.h"



class SelfOrganising {

	public:
		SelfOrganising();
		void setup();
		void update();
		void draw();

		Tree addTree(ofVec3f pos, ofVec3f dir, int idx);
		void basipetalPass();
		float recurBasipetally(shared_ptr<Metamer> metamer, int treeIdx);
		void acropetalPass();
		void recurAcropetally(shared_ptr<Metamer> metamer, float v, int treeIdx);
		void addNewTerminalShoot(shared_ptr<Metamer> metamer, float v, int treeIdx);
		void addNewAxillaryShoot(shared_ptr<Metamer> metamer, float v, int treeIdx);
		void addNewShoot(shared_ptr<Metamer> metamer, float v, ofVec3f defaultDir, ofVec3f growthDir, bool isTerminal, int treeIdx);
        shared_ptr<Metamer> addMetamer(shared_ptr<Metamer> parent_metamer, ofVec3f pos, ofVec3f dir, bool isTerminal, int treeIdx);

		SpaceColonization environment;

		vector<ofVec3f> metamerPositions;
		vector<glm::vec3> metamerVertices;
		vector<unsigned int> metamerIndices;
		vector<ofFloatColor> metamerColors;
        vector<float> metamerWidths;
		vector<ofVec3f> budAttractions;
		int metamerIdx;

		vector<ofVec3f> markers;
		vector<Tree> trees;
		vector<shared_ptr<Metamer>> metamersWithBuds;
		vector<ofVec3f> budPositions;

		ofVbo metamerVbo;

		float maxPerceptionFactor;

        ofShader shader;
		ofEasyCam cam;
		float mapSize;
};
