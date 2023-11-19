#include "spacecolonization.h"

SpaceColonization::SpaceColonization()
	: markerHash(markers)
{ }

int SpaceColonization::getWidth() {
    return width;
}

int SpaceColonization::getHeight() {
    return height;
}

int SpaceColonization::getDepth() {
    return depth;
}

void SpaceColonization::setup(int width, int height, int depth) {
	// string markerspawn = "random";

	int nummarkers = 10000;
	for (int i = 0; i < nummarkers; i++) {
		markers.push_back(ofVec3f(ofRandom(0, width), ofRandom(0, height), ofRandom(0, depth)));
		markerClosestBudDist.push_back(std::numeric_limits<float>::max());
		markerClosestBudIdx.push_back(-1);
	}

	// if (markerspawn == "trees") {
	// 	// TODO use envelope mthod of creating markers
	// 	int rootmarkers = 1000;
	// 	for (int i = 0; i < rootmarkers; i++) {
	// 		markers.push_back(ofVec3f(ofRandom(0, width), ofRandom(0, height), ofRandom(0, depth)));
	// 	}

	// 	int skymarkers = 1000;
	// 	for (int i = 0; i < skymarkers; i++) {
	// 		markers.push_back(ofVec3f(ofRandom(-mapSize/2, mapSize/2), ofRandom(mapSize/3, mapSize/2), ofRandom(-mapSize/2, mapSize/2)));
	// 	}
	// } else if (markerspawn == "random") {
	// 	// randomly create markers
		
	// } else if (markerspawn == "random2d") {
	// 	int nummarkers = 10000;
	// 	for (int i = 0; i < nummarkers; i++) {
	// 		markers.push_back(ofVec3f(ofRandom(-mapSize/2, mapSize/2), 0., ofRandom(-mapSize/2, mapSize/2)));
	// 	}
	// }
}

void SpaceColonization::setMaxPerceptionFactor(float _maxPerceptionFactor) {
	maxPerceptionFactor = _maxPerceptionFactor;
}

void SpaceColonization::updateBudEnvironment(vector<Tree> trees) {
	
	for (int i = 0; i < markers.size(); i++) {
		markerClosestBudDist[i] = std::numeric_limits<float>::max();
		markerClosestBudIdx[i] = -1;
	}
	markerHash.buildIndex();

	ofx::KDTree<ofVec3f>::SearchResults markerSearchResults;
	for (int i = 0; i < metamersWithBuds.size(); i++) {
		markerSearchResults.clear();

		// max number of buds to look for in radius
		int markerSearchSize = 100;
		markerSearchResults.resize(markerSearchSize);

		shared_ptr<Metamer> metamer = metamersWithBuds[i];
		metamer->terminalGrowthDirection = ofVec3f(0., 0., 0.);
		metamer->axillaryGrowthDirection = ofVec3f(0., 0., 0.);

		Tree tree = trees[metamer->treeIdx];

		float perceptionAngle = tree.perceptionAngle;
		float perceptionFactor = tree.perceptionFactor;
		float occupancyFactor = tree.occupancyFactor;

		// max radius to search for buds
		float budSearchRadius = metamer->length * perceptionFactor;
		int numMarkers = markerHash.findPointsWithinRadius(metamer->pos, budSearchRadius, markerSearchResults);

		if (numMarkers < 1) {
			continue;
		}

		for (int j = 0; j < numMarkers; j++) {
			ofVec3f marker = markers[markerSearchResults[j].first];
			float markerDistSq = markerSearchResults[j].second;

			if (markerDistSq < std::pow(metamer->length * occupancyFactor, 2.)) {
				markers.erase(markers.begin() + markerSearchResults[j].first);
			}
			
			ofVec3f markerDir = (marker - metamer->pos).normalize();
			float angle = markerDir.angleRad(metamer->direction);
			float axillaryAngle = markerDir.angleRad(metamer->axillaryDirection);
			if (metamer->axillary == NULL && axillaryAngle < angle && axillaryAngle < perceptionAngle) {
				if (markerDistSq < markerClosestBudDist[markerSearchResults[j].first]) {
					markerClosestBudDist[markerSearchResults[j].first] = markerDistSq;
					markerClosestBudIdx[markerSearchResults[j].first] = i;
				}
			} else if (metamer->terminal == NULL && angle < axillaryAngle && angle < perceptionAngle) {
				if (markerDistSq < markerClosestBudDist[markerSearchResults[j].first]) {
					markerClosestBudDist[markerSearchResults[j].first] = markerDistSq;
					markerClosestBudIdx[markerSearchResults[j].first] = i;
				}
			}
		}
	}
}

void SpaceColonization::updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree) {
	ofx::KDTree<ofVec3f>::SearchResults markerSearchResults;
	markerSearchResults.clear();

	// max number of buds to look for in radius
	int markerSearchSize = 100;
	markerSearchResults.resize(markerSearchSize);

	float perceptionAngle = tree.perceptionAngle;
	float perceptionFactor = tree.perceptionFactor;

	// max radius to search for buds
	float budSearchRadius = metamer->length * perceptionFactor;
	int numMarkers = markerHash.findPointsWithinRadius(metamer->pos, budSearchRadius, markerSearchResults);

	if (numMarkers < 1) {
		metamer->axillaryGrowthDirection = metamer->axillaryDirection;
		metamer->terminalGrowthDirection = metamer->direction;
	}

	for (int j = 0; j < numMarkers; j++) {
		int metamerIdx = markerClosestBudIdx[markerSearchResults[j].first];
		ofVec3f marker = markers[markerSearchResults[j].first];
		if (metamerIdx == metamer->idx) {
			ofVec3f markerDir = (marker - metamer->pos).normalize();
			float angle = markerDir.angleRad(metamer->direction);
			float axillaryAngle = markerDir.angleRad(metamer->axillaryDirection);
			if (metamer->axillary == NULL && axillaryAngle < angle && axillaryAngle < perceptionAngle) {
				metamer->axillaryGrowthDirection += markerDir;
			} else if (metamer->terminal == NULL && angle < axillaryAngle && angle < perceptionAngle) {
				metamer->terminalGrowthDirection += markerDir;
			}
		}
	}

	if (numMarkers > 0) {
		metamer->axillaryGrowthDirection.normalize();
		metamer->terminalGrowthDirection.normalize();
	}
}

void SpaceColonization::addToEnvironment(shared_ptr<Metamer> metamer) {
    metamersWithBuds.push_back(metamer);
	budPositions.push_back(metamer->pos);
}

void SpaceColonization::removeFromEnvironment(shared_ptr<Metamer> metamer) {
    for (int i = 0; i < metamersWithBuds.size(); i++) {
		if (metamersWithBuds[i]->idx == metamer->idx) {
			metamersWithBuds.erase(metamersWithBuds.begin() + i);
			budPositions.erase(budPositions.begin() + i);
			break;
		}
	}
}