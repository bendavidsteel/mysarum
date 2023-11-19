// #include "environment.h"
#include "trees.h"

class ShadowPropagation {//} : public Environment {
    public:
        void setup(float mapSize);
        void updateBudEnvironment();
        void updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree);
        void addToEnvironment(shared_ptr<Metamer> metamer);
        void removeFromEnvironment(shared_ptr<Metamer> metamer);

        int getWidth();
        int getHeight();
        int getDepth();

    private:
        vector<float> shadows;
        float C;
        float a;
        float b;
        int qmax;
        float shadowDist;

        int width;
        int height;
        int depth;
};