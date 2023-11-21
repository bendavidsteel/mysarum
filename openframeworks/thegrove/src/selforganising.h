#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"
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
		void setup(int _numBins, int _width, int _height, int _depth);
		void update(float windStrength, ofVec2f windDirection);
		void draw();
		void drawGui(int x, int y);

		vector<float> getNewMetamerHist();

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

		vector<float> spaceVector;
		float spaceVectorFactor;
		ofBufferObject spaceBuffer;

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
		float mapSize;
		int width, height, depth;

		int processStep;
		int basipetalCount, maxBasipetalCount;
		int acropetalCount, maxAcropetalCount;

		vector<float> newMetamerHist;
		int numBins;

		float windStrength;
		ofVec2f windDirection;
};
