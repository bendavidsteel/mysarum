#pragma once

#include "ofMain.h"

class ReactionDiffusion {
    public:
        void setup(int mapWidth, int mapHeight);
        void update(
            int mapFactor, int mapWidth, int mapHeight, 
            float time, float deltaTime, float timeOfDay, float timeOfMonth, 
            int cvDownScale,
            ofFbo flowFbo, ofTexture opticalFlowTexture
        );
        void draw();

        ofFbo getReactionFbo();

        void reSpawnReaction();
		void setReactionFeedRange(float feedMin, float feedRange);
		void setReactionFlowMag(float reactionFlowMag);

        void reloadShaders();

        void exit();

        void copyVariables();
        void moveToVariables();

    private:
		ofShader compute_reaction;

		ofFbo reactionFbo;
		ofFbo lastReactionFbo;

        float feedMin;
		float newFeedMin;
		float feedRange;
		float newFeedRange;

        float reactionFlowMag;
		float newReactionFlowMag;

        int bReSpawnReaction;
};