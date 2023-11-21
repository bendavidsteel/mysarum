#pragma once

#include "ofMain.h"

class ReactionDiffusion {
    public:
        void setup(int mapWidth, int mapHeight);
        void update();
        void draw();

        ofFbo getReactionFbo();

        void reloadShaders();

        void exit();

    private:
		ofShader compute_reaction;
        ofShader plane_renderer;
        ofPlanePrimitive plane;

		ofFbo reactionFbo;
		ofFbo lastReactionFbo;

        float feedMin;
		float newFeedMin;
		float feedRange;
		float newFeedRange;

        float reactionFlowMag;
		float newReactionFlowMag;

        int bReSpawnReaction;

        int mapWidth, mapHeight;
        glm::vec3 colourA, colourB;
};