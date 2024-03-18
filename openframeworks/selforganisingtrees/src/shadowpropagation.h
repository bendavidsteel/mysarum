// #include "environment.h"
#include "trees.h"

class ShadowPropagation {//} : public Environment {
    public:
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