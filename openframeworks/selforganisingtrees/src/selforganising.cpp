#include "selforganising.h"
// #include "test.cuh"
#include "ofConstants.h"

#define MAX_METAMERS 1000000
#define PI 3.14159265;

SelfOrganising::SelfOrganising()
{
}

//--------------------------------------------------------------
void SelfOrganising::setup(){
	mapSize = 200.;
	maxAcropetalCount = 100;
	maxBasipetalCount = 100;

	// create node vbo
	metamerVertices.reserve(MAX_METAMERS * 2);
	metamerIndices.reserve(MAX_METAMERS * 2);
	metamerColors.reserve(MAX_METAMERS * 2);
	metamerWidths.reserve(MAX_METAMERS * 2);

	string markerspawn = "tree";

	int width = 100;
	int height = 100;
	int depth = 100;
	environment.setup(width, height, depth);

	if (markerspawn == "trees") {
		// create initial nodes
		int numInitialNodes = int(ofRandom(2, 5));
		for (int i = 0; i < numInitialNodes; i++) {
			ofVec3f pos = ofVec3f(ofRandom(0., environment.getWidth()), 0., ofRandom(0., environment.getDepth()));
			ofVec3f dir = ofVec3f(0., 1., 0.);
			Tree tree = addTree(pos, dir, i);
			trees.push_back(tree);
		}
	} else if (markerspawn == "tree") {
		ofVec3f pos = ofVec3f(environment.getWidth() / 2., 0., environment.getDepth() / 2.);
		ofVec3f dir = ofVec3f(0., 1., 0.);
		Tree tree = addTree(pos, dir, 0);
		trees.push_back(tree);
	
	} else if (markerspawn == "random") {
		// create initial nodes
		int numInitialNodes = int(ofRandom(3, 8));
		for (int i = 0; i < numInitialNodes; i++) {
			ofVec3f pos = ofVec3f(ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2));
			ofVec3f dir = ofVec3f(ofRandom(-1., 1.), ofRandom(-1., 1.), ofRandom(-1., 1.)).normalize();
			Tree tree = addTree(pos, dir, i);
			trees.push_back(tree);
		}
	} else if (markerspawn == "random2d") {
		// create initial nodes
		int numInitialNodes = int(ofRandom(3, 8));
		for (int i = 0; i < numInitialNodes; i++) {
			ofVec3f pos = ofVec3f(ofRandom(-mapSize/2, mapSize/2), 0., ofRandom(-mapSize/2, mapSize/2));
			ofVec3f dir = ofVec3f(ofRandom(-1., 1.), 0., ofRandom(-1., 1.)).normalize();
			Tree tree = addTree(pos, dir, i);
			trees.push_back(tree);
		}
	}

	shader.load("node.vert", "node.frag", "node.geom");

	cam.setFarClip(ofGetWidth()*10);
	cam.setNearClip(0.1);

	metamerVbo.setVertexData(metamerVertices.data(), MAX_METAMERS * 2, GL_DYNAMIC_DRAW);
	metamerVbo.setIndexData(metamerIndices.data(), MAX_METAMERS * 2, GL_DYNAMIC_DRAW);
	metamerVbo.setColorData(metamerColors.data(), MAX_METAMERS * 2, GL_DYNAMIC_DRAW);
	
	shader.begin();
	int widthAttLoc = shader.getAttributeLocation("width");
	metamerVbo.setAttributeData(widthAttLoc, metamerWidths.data(), 1, MAX_METAMERS * 2, GL_DYNAMIC_DRAW);
	shader.end();
}

Tree SelfOrganising::addTree(ofVec3f startPos, ofVec3f dir, int treeIdx) {
	if (metamerIdx >= MAX_METAMERS) {
		throw std::runtime_error("Maximum number of metamers reached");
	}

	Tree tree;
	tree.initialWidth = 0.1;
	tree.axillaryAngle = 0.3 * PI;
	tree.baseLength = 2.;

	shared_ptr<Metamer> newMetamer(new Metamer());

	int idx = metamerIdx;
	newMetamer->idx = idx;
	metamerIdx++;

	ofVec3f pos = startPos + (dir * tree.baseLength);
	newMetamer->treeIdx = idx;
	newMetamer->length = tree.baseLength;
	newMetamer->pos = pos;
	newMetamer->direction = dir;
	newMetamer->width = tree.initialWidth;

	ofVec3f randomVec = ofVec3f(ofRandom(-1., 1.), ofRandom(-1., 1.), ofRandom(-1., 1.)).normalize();
	ofVec3f randomOrth = randomVec.cross(dir).normalize();
	newMetamer->axillaryDirection = dir.rotateRad(tree.axillaryAngle, randomOrth);

	tree.root = newMetamer;
	tree.alpha = 2.;
	tree.lambda = 0.48;
	tree.occupancyFactor = 2.;
	tree.perceptionFactor = 5.;
	tree.perceptionAngle = 0.3 * PI;
	tree.idx = treeIdx;
	tree.growthWeight = 0.3;
	tree.defaultWeight = 0.3;
	tree.tropismWeight = 0.1;
	tree.tropismDir = ofVec3f(0., -1., 0.);
	tree.branchExp = 3.;
	tree.dropRatio = 0.;
	tree.v = 0.;

	metamerVertices[2 * idx] = glm::vec3(startPos.x, startPos.y, startPos.z);
	metamerVertices[(2 * idx) + 1] = glm::vec3(pos.x, pos.y, pos.z);
	metamerIndices[2 * idx] = 2 * idx;
	metamerIndices[(2 * idx) + 1] = (2 * idx) + 1;
	metamerColors[2 * idx] = ofFloatColor(1, 1, 1, 1);
	metamerColors[(2 * idx) + 1] = ofFloatColor(1, 1, 1, 1);
    metamerWidths[2 * idx] = tree.initialWidth;
    metamerWidths[(2 * idx) + 1] = tree.initialWidth;

	environment.addToEnvironment(newMetamer);

	return tree;
}


void SelfOrganising::startBasipetalPass() {
    for (unsigned int i = 0; i < trees.size(); i++) {
        Tree & tree = trees[i];
        basipetalStack.push({tree.root, tree.idx, false});
	}
}

void SelfOrganising::basipetalPass() {
	basipetalCount = 0;
	while (!basipetalStack.empty() && basipetalCount < maxBasipetalCount) {
		BasipetalState & state = basipetalStack.top();

		if (state.metamer->dropped) {
			basipetalStack.pop();
			continue;
		}

		if (!state.processedChildren) {
			state.processedChildren = true;
			bool pushedChild = false;

			if (state.metamer->terminal != NULL) {
				basipetalStack.push({state.metamer->terminal, state.treeIdx, false});
				pushedChild = true;
			}

			if (state.metamer->axillary != NULL) {
				basipetalStack.push({state.metamer->axillary, state.treeIdx, false});
				pushedChild = true;
			}

			if (pushedChild) {
				continue;
			}
		}

		bool updateBud = state.metamer->terminal == NULL || state.metamer->axillary == NULL;
		if (updateBud) {
			environment.updateBudEnvironment(state.metamer, trees[state.treeIdx]);
		}

		// Process the current metamer
		float sumWidthSq = 0;
		if (state.metamer->terminal != NULL) {
			state.metamer->terminalQ = state.metamer->terminal->terminalQ + state.metamer->terminal->axillaryQ;
			sumWidthSq += std::pow(state.metamer->terminal->width, trees[state.treeIdx].branchExp);
		}

		if (state.metamer->axillary != NULL) {
			state.metamer->axillaryQ = state.metamer->axillary->terminalQ + state.metamer->axillary->axillaryQ;
			sumWidthSq += std::pow(state.metamer->axillary->width, trees[state.treeIdx].branchExp);
		}

		if (sumWidthSq > 0) {
			state.metamer->width = std::pow(sumWidthSq, 1. / trees[state.treeIdx].branchExp);
		} else {
			float initialWidth = trees[state.treeIdx].initialWidth;
			state.metamer->width = initialWidth;
		}

		metamerWidths[2*state.metamer->idx] = state.metamer->width;
		if (state.metamer->parent != NULL) {
			metamerWidths[(2*state.metamer->idx) + 1] = state.metamer->parent->width;
		} else {
			metamerWidths[(2*state.metamer->idx) + 1] = state.metamer->width;
		}

		// assess whether to drop branch
		float q = state.metamer->terminalQ + state.metamer->axillaryQ;
		if (q / state.metamer->width < trees[state.treeIdx].dropRatio) {
			state.metamer->toDrop = true;
		}

		if (state.metamer->parent == NULL) {
			// This is the root metamer
			Tree & tree = trees[state.treeIdx];
			tree.v = tree.alpha * (state.metamer->terminalQ + state.metamer->axillaryQ);
		}

		basipetalStack.pop();

		basipetalCount++;
	}
}

void SelfOrganising::startAcropetalPass() {
	for (unsigned int i = 0; i < trees.size(); i++) {
        Tree tree = trees[i];
        acropetalStack.push({tree.root, tree.v, i});
	}
}

void SelfOrganising::acropetalPass() {
	acropetalCount = 0;
	while (!acropetalStack.empty() && acropetalCount < maxAcropetalCount) {
		AcropetalState state = acropetalStack.top();
		acropetalStack.pop();

		acropetalCount++;

		if (state.metamer->toDrop) {
			environment.removeFromEnvironment(state.metamer);
			state.metamer->toDrop = false;
			state.metamer->dropped = true;

			metamerColors[2 * state.metamer->idx] = ofFloatColor(0, 0, 0, 0);
			metamerColors[(2 * state.metamer->idx) + 1] = ofFloatColor(0, 0, 0, 0);

			if (state.metamer->terminal != NULL) {
				state.metamer->terminal->toDrop = true;
				acropetalStack.push({state.metamer->terminal, 0., state.treeIdx});
			}

			if (state.metamer->axillary != NULL) {
				state.metamer->axillary->toDrop = true;
				acropetalStack.push({state.metamer->axillary, 0., state.treeIdx});
			}

			continue;
		}

		if (state.metamer->dropped) {
			continue;
		}

		float lambda = trees[state.treeIdx].lambda;
		float lQm = lambda * state.metamer->terminalQ;
		float lQl = (1 - lambda) * state.metamer->axillaryQ;
		float denom = lQm + lQl;
		float vm = state.v * (lQm / denom);
		float vl = state.v * (lQl / denom);

		// Process terminal
		if (state.metamer->terminal != NULL) {
			acropetalStack.push({state.metamer->terminal, vm, state.treeIdx});
		} else {
			if (vm >= 1.) {
				addNewTerminalShoot(state.metamer, vm, state.treeIdx);
			}
		}

		// Process axillary
		if (state.metamer->axillary != NULL) {
			acropetalStack.push({state.metamer->axillary, vl, state.treeIdx});
		} else {
			if (vl >= 1.) {
				addNewAxillaryShoot(state.metamer, vl, state.treeIdx);
			}
		}
	}
}


void SelfOrganising::addNewTerminalShoot(shared_ptr<Metamer> metamer, float v, int treeIdx) {
	ofVec3f terminalDir = metamer->direction;
	ofVec3f growthDir = metamer->terminalGrowthDirection;
	addNewShoot(metamer, v, terminalDir, growthDir, true, treeIdx);
	
}

void SelfOrganising::addNewAxillaryShoot(shared_ptr<Metamer> metamer, float v, int treeIdx) {
	ofVec3f axillaryDir = metamer->axillaryDirection;
	ofVec3f growthDir = metamer->axillaryGrowthDirection;
	addNewShoot(metamer, v, axillaryDir, growthDir, false, treeIdx);
}

void SelfOrganising::addNewShoot(shared_ptr<Metamer> metamer, float v, ofVec3f defaultDir, ofVec3f growthDir, bool isTerminal, int treeIdx) {
	Tree tree = trees[treeIdx];
	int n = 1;//std::floor(v);
	float l = 1;//v / float(n);

	if (n > 0) {
		if (metamer->terminal != NULL && metamer->axillary != NULL) {
			environment.removeFromEnvironment(metamer);
		}
	}

	vector<shared_ptr<Metamer>> metamerShoot;
	metamerShoot.push_back(metamer);

	for (int i = 0; i < n; i++) {
		shared_ptr<Metamer> baseMetamer = metamerShoot.back();
		ofVec3f dir = (tree.defaultWeight * defaultDir + tree.growthWeight * growthDir + tree.tropismWeight * tree.tropismDir).normalize();
		ofVec3f pos = baseMetamer->pos + (l * tree.baseLength * dir);
		shared_ptr<Metamer> newMetamer = addMetamer(baseMetamer, pos, dir, isTerminal, treeIdx);
		metamerShoot.push_back(newMetamer);
		isTerminal = true;
	}
}

shared_ptr<Metamer> SelfOrganising::addMetamer(shared_ptr<Metamer> parent_metamer, ofVec3f pos, ofVec3f dir, bool isTerminal, int treeIdx) {
	if (metamerIdx >= MAX_METAMERS) {
		throw std::runtime_error("Maximum number of metamers reached");
	}

	shared_ptr<Metamer> newMetamer(new Metamer());

	if (isTerminal) {
		parent_metamer->terminal = newMetamer;
	} else {
		parent_metamer->axillary = newMetamer;
	}

	newMetamer->parent = parent_metamer;

	int idx = metamerIdx;
	newMetamer->idx = idx;

	Tree tree = trees[treeIdx];
	ofVec3f startPos = parent_metamer->pos;

	newMetamer->treeIdx = treeIdx;
	newMetamer->pos = pos;
	newMetamer->direction = dir;
	newMetamer->length = (pos - startPos).length();
	newMetamer->width = tree.initialWidth;

	ofVec3f randomVec = ofVec3f(ofRandom(-1., 1.), ofRandom(-1., 1.), ofRandom(-1., 1.)).normalize();
	ofVec3f randomOrth = randomVec.cross(dir).normalize();
	newMetamer->axillaryDirection = dir.rotateRad(tree.axillaryAngle, randomOrth);

	metamerVertices[2 * idx] = glm::vec3(startPos.x, startPos.y, startPos.z);
	metamerVertices[(2 * idx) + 1] = glm::vec3(pos.x, pos.y, pos.z);
	metamerIndices[2 * idx] = 2 * idx;
	metamerIndices[(2 * idx) + 1] = (2 * idx) + 1;
	metamerColors[2 * idx] = ofFloatColor(1, 1, 1, 1);
	metamerColors[(2 * idx) + 1] = ofFloatColor(1, 1, 1, 1);
    metamerWidths[2 * idx] = tree.initialWidth;
    metamerWidths[(2 * idx) + 1] = tree.initialWidth;

	metamerIdx++;

	environment.addToEnvironment(newMetamer);

	return newMetamer;
}


void SelfOrganising::update() {
	if (metamerIdx < 10000) {
		if (processStep == 0) {
			if (environment.getUpdateCount() == 0) {
				environment.startUpdateBudEnvironment();
			}
			environment.updateBudEnvironment(trees);
			if (environment.getUpdateCount() == 0) {
				processStep = 1;
			}
		} else if (processStep == 1) {
			if (basipetalStack.empty()) {
				startBasipetalPass();
			}
			basipetalPass();
			if (basipetalStack.empty()) {
				processStep = 2;
			}
		} else if (processStep == 2) {
			if (acropetalStack.empty()) {
				startAcropetalPass();
			}
			acropetalPass();
			if (acropetalStack.empty()) {
				processStep = 0;
			}
			bool metamerAdded = true;
			if (metamerAdded) {
				metamerVbo.updateVertexData(metamerVertices.data(), MAX_METAMERS * 2);
				metamerVbo.updateIndexData(metamerIndices.data(), MAX_METAMERS * 2);
				metamerVbo.updateColorData(metamerColors.data(), MAX_METAMERS * 2);

				shader.begin();
				int widthAttLoc = shader.getAttributeLocation("width");
				metamerVbo.updateAttributeData(widthAttLoc, metamerWidths.data(), MAX_METAMERS * 2);
				shader.end();
			}
		}
	}
}

//--------------------------------------------------------------
void SelfOrganising::draw() {
	// draw nodes
	ofSetColor(255);

	float centreX = environment.getWidth() / 2.;
	float centreY = 100.;
	float centreZ = environment.getDepth() / 2.;
	cam.lookAt(glm::vec3(centreX, centreY, centreZ));

	// float timeOfDay = ofGetElapsedTimef() * 0.1;
	float camDist = 500;
	float camX = centreX;// + (camDist * std::sin(timeOfDay));
	float camY = centreY + 100;// * std::sin(timeOfDay / 3));
	float camZ = centreZ + 200;// * std::cos(timeOfDay));
	cam.setPosition(camX, camY, camZ);

	cam.begin();

	// ofRotateXDeg(15);

	// ofSetColor(0);
	ofDrawGrid(20, 10, false, false, true, false);

	shader.begin();
	// // just draw a simple line
	// ofVbo vbo;
	// vector<glm::vec3> vertices;
	// vertices.push_back(glm::vec3(0, 0, 0));
	// vertices.push_back(glm::vec3(100, 0, 0));
	// vector<unsigned int> indices;
	// indices.push_back(0);
	// indices.push_back(1);
	// vector<ofFloatColor> colors;
	// colors.push_back(ofFloatColor(1, 1, 1, 1));
	// colors.push_back(ofFloatColor(1, 1, 1, 1));
	// vbo.setVertexData(vertices.data(), 2, GL_DYNAMIC_DRAW);
	// vbo.setIndexData(indices.data(), 2, GL_DYNAMIC_DRAW);
	// vbo.setColorData(colors.data(), 2, GL_DYNAMIC_DRAW);
	// vbo.drawElements(GL_LINES, 2);

	metamerVbo.drawElements(GL_LINES, metamerVbo.getNumIndices());
	
	shader.end();

	// draw markers
	// bool drawMarkers = true;
	// if (drawMarkers)
	// {
	// 	ofSetColor(255, 0, 0);
	// 	for (int i = 0; i < markers.size(); i++) {
	// 		ofDrawCircle(markers[i], 1);
	// 	}
	// }

	cam.end();
}
