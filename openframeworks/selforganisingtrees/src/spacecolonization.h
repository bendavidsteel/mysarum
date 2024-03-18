#include "ofMain.h"
#include "ofxSpatialHash.h"
#include "trees.h"

class SpaceColonization {
    public:
        SpaceColonization();
        void setup(int width, int height, int depth);
        void startUpdateBudEnvironment();
        void updateBudEnvironment(vector<Tree> trees);
        void updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree);
        void addToEnvironment(shared_ptr<Metamer> metamer);
        void removeFromEnvironment(shared_ptr<Metamer> metamer);

        int getUpdateCount();

        int getWidth();
        int getHeight();
        int getDepth();
        
    private:
        vector<ofVec3f> budPositions;
        vector<shared_ptr<Metamer>> metamersWithBuds;

        ofx::KDTree<ofVec3f> markerHash;
        vector<ofVec3f> markers;
        vector<float> markerClosestBudDist;
        vector<int> markerClosestBudIdx;

        float maxPerceptionFactor;
        int width, height, depth;

        int updateBudCount;
        bool markerRemoved;
};