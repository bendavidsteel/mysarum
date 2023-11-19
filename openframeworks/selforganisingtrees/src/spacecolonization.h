#include "ofMain.h"
#include "ofxSpatialHash.h"
#include "trees.h"

class SpaceColonization {
    public:
        SpaceColonization();
        void setup(int width, int height, int depth);
        void updateBudEnvironment(vector<Tree> trees);
        void updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree);
        void addToEnvironment(shared_ptr<Metamer> metamer);
        void removeFromEnvironment(shared_ptr<Metamer> metamer);

        int getWidth();
        int getHeight();
        int getDepth();

        void setMaxPerceptionFactor(float maxPerceptionFactor);
    private:
        vector<ofVec3f> budPositions;
        vector<shared_ptr<Metamer>> metamersWithBuds;

        ofx::KDTree<ofVec3f> markerHash;
        vector<ofVec3f> markers;
        vector<float> markerClosestBudDist;
        vector<int> markerClosestBudIdx;

        float maxPerceptionFactor;
        int width, height, depth;
};