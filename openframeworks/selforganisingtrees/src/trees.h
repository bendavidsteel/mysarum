#pragma once

#include "ofMain.h"

struct Metamer {
	shared_ptr<Metamer> terminal;
	ofVec3f terminalGrowthDirection;
	float terminalQ;
	shared_ptr<Metamer> axillary;
	ofVec3f axillaryDirection;
	ofVec3f axillaryGrowthDirection;
	float axillaryQ;
	int idx;
	int treeIdx;
	float width;
	ofVec3f pos;
	ofVec3f direction;
	float length;
};

struct Tree {
	int idx;
	float growthWeight;
	float defaultWeight;
	float tropismWeight;
	ofVec3f tropismDir;
	float initialWidth;
	float baseLength;
	float axillaryAngle;
	float lambda;
	float v;
	float alpha;
	float perceptionAngle;
	float perceptionFactor;
	float occupancyFactor;
	shared_ptr<Metamer> root;
};