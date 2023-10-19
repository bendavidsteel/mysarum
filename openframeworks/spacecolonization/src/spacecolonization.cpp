#include "spacecolonization.h"
// #include "test.cuh"
#include "ofConstants.h"

#define MAX_NODES 1000000

SpaceColonization::SpaceColonization():
	nodeHash(nodePositions),
	attractorHash(attractors)
{
}

//--------------------------------------------------------------
void SpaceColonization::setup(){

	attractionDistance = ofGetWidth() / 4;
	killDistance = ofGetWidth() / 1000;
	segmentLength = 1.;
	initialWidth = 1.;
	maxWidth = 5.;
	growthFactor = 0.0002;

	int mapSize = 1000;

	// randomly create attractors
	int numAttractors = 10000;
	for (int i = 0; i < numAttractors; i++) {
		attractors.push_back(ofVec3f(ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2)));
	}

	shader.load("node.vert", "node.frag", "node.geom");

	cam.setFarClip(ofGetWidth()*10);
	cam.setNearClip(0.1);

	// create node vbo
	nodeVertices.reserve(MAX_NODES * 2);
	nodeIndices.reserve(MAX_NODES * 2);
	nodeColors.reserve(MAX_NODES * 2);
	nodeWidths.reserve(MAX_NODES * 2);

	// create initial nodes
	int numInitialNodes = int(ofRandom(3, 8));
	for (int i = 0; i < numInitialNodes; i++) {
		ofVec3f pos = ofVec3f(ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2), ofRandom(-mapSize/2, mapSize/2));
		addNode(pos, pos, i);
	}
	nodeIdx = numInitialNodes;

	nodeVbo.setVertexData(nodeVertices.data(), MAX_NODES * 2, GL_DYNAMIC_DRAW);
	nodeVbo.setIndexData(nodeIndices.data(), MAX_NODES * 2, GL_DYNAMIC_DRAW);
	nodeVbo.setColorData(nodeColors.data(), MAX_NODES * 2, GL_DYNAMIC_DRAW);
	
	shader.begin();
	int widthAttLoc = shader.getAttributeLocation("width");
	nodeVbo.setAttributeData(widthAttLoc, nodeWidths.data(), 1, MAX_NODES * 2, GL_DYNAMIC_DRAW);
	shader.end();
}

//--------------------------------------------------------------
void SpaceColonization::update(){
	// construct node spatial hash tree
	nodeHash.buildIndex();
	ofx::KDTree<ofVec3f>::SearchResults nodeSearchResults;

	// loop through attractors and find nearest node, and add attractor force
	for (int i = 0; i < attractors.size(); i++) {
		nodeSearchResults.clear();
		nodeSearchResults.resize(1);

		nodeHash.findNClosestPoints(attractors[i], 1, nodeSearchResults);
		int closestNode = nodeSearchResults[0].first;
		nodeAttractions[closestNode] += (attractors[i] - nodePositions[closestNode]).normalize();
	}

	// Wrapper::wrapper();

	// create attractor spatial hash
	attractorHash.buildIndex();
	ofx::KDTree<ofVec3f>::SearchResults attractorSearchResults;

	// loop through nodes with attraction, and add a node in the direction of the attraction
	bool nodeAdded = false;
	int numNodes = nodePositions.size();
	for (int i = 0; i < numNodes; i++) {
		if (nodeAttractions[i].length() > 0) {
			nodeAdded = true;
			ofVec3f orgPos = nodePositions[i];
			ofVec3f newPos = nodePositions[i] + nodeAttractions[i].normalize() * segmentLength;

			addNode(orgPos, newPos, nodeIdx, nodeTree[i]);
			nodeIdx++;

			// check if new node is close to an attractor, and remove attractor if so
			attractorSearchResults.clear();
			attractorSearchResults.resize(1);

			attractorHash.findPointsWithinRadius(nodePositions.back(), killDistance, attractorSearchResults);
			if (attractorSearchResults.size() > 0) {
				attractors.erase(attractors.begin() + attractorSearchResults[0].first);
			}

			nodeAttractions[i] = ofVec3f(0, 0, 0);
		}
	}

	if (nodeAdded) {
		nodeVbo.updateVertexData(nodeVertices.data(), MAX_NODES * 2);
		nodeVbo.updateIndexData(nodeIndices.data(), MAX_NODES * 2);
		nodeVbo.updateColorData(nodeColors.data(), MAX_NODES * 2);

		shader.begin();
		int widthAttLoc = shader.getAttributeLocation("width");
		nodeVbo.updateAttributeData(widthAttLoc, nodeWidths.data(), MAX_NODES * 2);
		shader.end();
	}
}

//--------------------------------------------------------------
void SpaceColonization::draw() {
	// draw nodes
	ofSetColor(255);

	float centreX = 0.;
	float centreY = 0.;
	float centreZ = 0.;
	cam.lookAt(glm::vec3(centreX, centreY, centreZ));

	float timeOfDay = ofGetElapsedTimef() * 0.1;
	float camDist = 1000.;
	float camX = centreX + (camDist * std::sin(timeOfDay));
	float camY = centreY + (camDist * std::sin(timeOfDay / 3));
	float camZ = centreZ + (camDist * std::cos(timeOfDay));
	cam.setPosition(camX, camY, camZ);

	cam.begin();

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
	nodeVbo.drawElements(GL_LINES, nodeVbo.getNumIndices());
	shader.end();

	cam.end();

	// draw attractors
	bool drawAttractors = false;
	if (drawAttractors)
	{
		ofSetColor(255, 0, 0);
		for (int i = 0; i < attractors.size(); i++) {
			ofDrawCircle(attractors[i], 1);
		}
	}
}

void SpaceColonization::addNode(ofVec3f posA, ofVec3f posB, int idx, shared_ptr<Node> parent) {
    addNodeData(posA, posB, idx);

    shared_ptr<Node> p(new Node);
	p->idx = idx;
	p->width = 1;
	p->opacity = 1;

	nodeTree.push_back(p);

    parent->children.push_back(p);
    p->parent = parent;
    recurWidth(p);
}

void SpaceColonization::addNode(ofVec3f posA, ofVec3f posB, int idx) {
    addNodeData(posA, posB, idx);

    shared_ptr<Node> p(new Node);
    p->idx = idx;
    p->width = initialWidth;
    p->opacity = 1;

    nodeTree.push_back(p);
    nodeRoots.push_back(p);
}

void SpaceColonization::addNodeData(ofVec3f posA, ofVec3f posB, int idx) {
	nodePositions.push_back(posB);
	nodeAttractions.push_back(ofVec3f(0, 0, 0));

	nodeVertices[2 * idx] = glm::vec3(posA.x, posA.y, posA.z);
	nodeVertices[(2 * idx) + 1] = glm::vec3(posB.x, posB.y, posB.z);
	nodeIndices[2 * idx] = 2 * idx;
	nodeIndices[(2 * idx) + 1] = (2 * idx) + 1;
	nodeColors[2 * idx] = ofFloatColor(1, 1, 1, 1);
	nodeColors[(2 * idx) + 1] = ofFloatColor(1, 1, 1, 1);
    nodeWidths[2 * idx] = initialWidth;
    nodeWidths[(2 * idx) + 1] = initialWidth;
}

void SpaceColonization::recurWidth(shared_ptr<Node> node) {
	// recur up the tree, adding widths
	if (node->parent != NULL) {
		node->parent->width = min(float(node->parent->width + growthFactor * node->width), maxWidth);
		nodeWidths[2 * node->parent->idx] = node->parent->width;
        nodeWidths[(2 * node->parent->idx) + 1] = node->parent->width;
		recurWidth(node->parent);
	}
}
