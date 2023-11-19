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
	// budHash.buildIndex();
    // ofx::KDTree<ofVec3f>::SearchResults budSearchResults;

	// loop through markers and find nearest node, and add attractor force
	// for (int i = 0; i < markers.size(); i++) {
	// 	budSearchResults.clear();
	// 	// max number of buds to look for in radius
	// 	int budSearchSize = 10;
	// 	budSearchResults.resize(budSearchSize);

	// 	// max radius to search for buds
	// 	float maxMetamerLength = 2.;
	// 	float budSearchRadius = maxMetamerLength * maxPerceptionFactor;
	// 	int numPoints = budHash.findPointsWithinRadius(markers[i], budSearchRadius, budSearchResults);

	// 	if (numPoints < 1) {
	// 		continue;
	// 	}

	// 	int closestBud = budSearchResults[0].first;
	// 	shared_ptr<Metamer> closestMetamer = metamersWithBuds[closestBud];
	// 	float closestBudDistSq = budSearchResults[0].second;
	// 	float occupancyFactor = trees[closestMetamer->treeIdx].occupancyFactor;

	// 	// check if in occupancy zone
	// 	if (closestBudDistSq < std::pow(closestMetamer->length * occupancyFactor, 2.)) {
	// 		markers.erase(markers.begin() + i);
	// 		i--;
	// 		continue;
	// 	}

	// 	for (int j = 0; j < numPoints; j++) {
	// 		shared_ptr<Metamer> metamer = metamersWithBuds[budSearchResults[j].first];
	// 		float budDistSq = budSearchResults[j].second;

	// 		float perceptionAngle = trees[metamer->treeIdx].perceptionAngle;
	// 		float perceptionFactor = trees[metamer->treeIdx].perceptionFactor;
			
	// 		if (budDistSq < std::pow(metamer->length * perceptionFactor, 2.)) {
	// 			ofVec3f markerDir = (markers[i] - metamer->pos).normalize();
	// 			float angle = markerDir.angle(metamer->direction);
	// 			float axillaryAngle = markerDir.angle(metamer->axillaryDirection);
	// 			if (axillaryAngle < angle && axillaryAngle < perceptionAngle) {
	// 				metamer->axillaryGrowthDirection += markerDir;
	// 			} else if (angle < axillaryAngle && angle < perceptionAngle) {
	// 				metamer->terminalGrowthDirection += markerDir;
	// 			}
	// 		}
	// 	}
	// }

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
			if (axillaryAngle < angle && axillaryAngle < perceptionAngle) {
				markerClosestBudDist[markerSearchResults[j].first] = markerDistSq;
				markerClosestBudIdx[markerSearchResults[j].first] = i;
				metamer->axillaryGrowthDirection += markerDir;
			} else if (angle < axillaryAngle && angle < perceptionAngle) {
				markerClosestBudDist[markerSearchResults[j].first] = markerDistSq;
				markerClosestBudIdx[markerSearchResults[j].first] = i;
				metamer->terminalGrowthDirection += markerDir;
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
			if (axillaryAngle < angle && axillaryAngle < perceptionAngle) {
				metamer->axillaryGrowthDirection += markerDir;
			} else if (angle < axillaryAngle && angle < perceptionAngle) {
				metamer->terminalGrowthDirection += markerDir;
			}
		}
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