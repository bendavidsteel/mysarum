#include "shadowpropagation.h"

void ShadowPropagation::setup(int _width, int _height, int _depth) {
    C = 1.;
    a = 0.5;
    b = 1.1;
    qmax = 18;

    width = _width;
    height = _height;
    depth = _depth;

    bool randomShadows = true;

    shadows.resize(width * height * depth);
    for (int i = 0; i < width * height * depth; i++) {
        if (randomShadows) {
            shadows[i] = ofRandom(0, C/3.);
        } else {
            shadows[i] = 0.;
        }
    }
}

void ShadowPropagation::startUpdateBudEnvironment() {
    return;
}

void ShadowPropagation::updateBudEnvironment(vector<Tree> trees) {
    return;
}

int ShadowPropagation::getUpdateCount() {
    return 0;
}

void ShadowPropagation::updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree) {
    metamer->terminalGrowthDirection = ofVec3f(0, 0, 0);
    metamer->axillaryGrowthDirection = ofVec3f(0, 0, 0);

    ofVec3f pos = metamer->pos;
    float s = shadows[int(pos.x) + (int(pos.y) * width) + (int(pos.z) * width * height)];
    float q = std::max(C - s + a, float(0.));
    metamer->terminalQ = q;
    metamer->axillaryQ = q;

    float minShadowTerminal = C;
    ofVec3f minShadowDirTerminal = ofVec3f(0, 0, 0);
    float minShadowAxillary = C;
    ofVec3f minShadowDirAxillary = ofVec3f(0, 0, 0);

    float perceptionDist = tree.perceptionFactor * metamer->length;
    int numPointsRadius = perceptionDist / shadowDist;
    for (int i = int(pos.x) - numPointsRadius; i < int(pos.x) + numPointsRadius; i++) {
        for (int j = int(pos.y) - numPointsRadius; j < int(pos.y) + numPointsRadius; j++) {
            for (int k = int(pos.z) - numPointsRadius; k < int(pos.z) + numPointsRadius; k++) {
                if (i < 0 || i >= width || j < 0 || j >= height || k < 0 || k >= depth) {
                    continue;
                }
                ofVec3f point(i, j, k);
                float dist = point.distance(pos);
                if (dist > perceptionDist) {
                    continue;
                }
                if (metamer->terminal != NULL) {
                    float angle = (point - pos).angleRad(metamer->direction);
                    if (angle < tree.perceptionAngle / 2.) {
                        float shadow = shadows[i + (j * width) + (k * width * height)];
                        if (shadow < minShadowTerminal) {
                            minShadowTerminal = shadow;
                            minShadowDirTerminal = point - pos;
                        };
                    }
                }

                if (metamer->axillary != NULL) {
                    float angle = (point - pos).angleRad(metamer->axillaryDirection);
                    if (angle < tree.perceptionAngle / 2.) {
                        float shadow = shadows[i + (j * width) + (k * width * height)];
                        if (shadow < minShadowAxillary) {
                            minShadowAxillary = shadow;
                            minShadowDirAxillary = point - pos;
                        };
                    }
                }
            }
        }
    }

    if (minShadowTerminal < C) {
        metamer->terminalGrowthDirection = minShadowDirTerminal;
    } else {
        metamer->terminalGrowthDirection = metamer->direction;
    }

    if (minShadowAxillary < C) {
        metamer->axillaryGrowthDirection = minShadowDirAxillary;
    } else {
        metamer->axillaryGrowthDirection = metamer->axillaryDirection;
    }
}

void ShadowPropagation::addToEnvironment(shared_ptr<Metamer> metamer) {
    ofVec3f pos = metamer->pos;
    for (int q = 0; q <= qmax; q++) {
        for (int i = -q; i <= q; i++) {
            for (int k = -q; k <= q; k++) {
                int x = min(max(int(pos.x) + i, 0), width - 1);
                int y = min(max(int(pos.y) - q, 0), height - 1);
                int z = min(max(int(pos.z) + k, 0), depth - 1);
                int idx = x + (y * width) + (z * width * height);
                float shadow = shadows[idx];
                shadow += a * std::pow(b, -q);
                shadows[idx] = shadow;
            }
        }
    }
}

void ShadowPropagation::removeFromEnvironment(shared_ptr<Metamer> metamer) {
    ofVec3f pos = metamer->pos;
    for (int q = 0; q <= qmax; q++) {
        for (int i = -q; i <= q; i++) {
            for (int k = -q; k <= q; k++) {
                int idx = int(pos.x + i) + int(pos.y - q) * width + int(pos.z + k) * width * height;
                float shadow = shadows[idx];
                shadow -= a * std::pow(b, -q);
                shadows[idx] = shadow;
            }
        }
    }
}

int ShadowPropagation::getWidth() {
    return width;
}

int ShadowPropagation::getHeight() {
    return height;
}

int ShadowPropagation::getDepth() {
    return depth;
}