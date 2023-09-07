#include "reactiondiffusion.h"

void ReactionDiffusion::setup(int mapWidth, int mapHeight) {
    feedMin = 0.01;
    feedRange = 0.09;
	bReSpawnReaction = 1;

    reactionFlowMag = 0.;

    // reaction diffusion setup
	reactionFbo.allocate(mapWidth, mapHeight, GL_RG16);
	lastReactionFbo.allocate(mapWidth, mapHeight, GL_RG16);

    // load reaction diffusion shaders
	compute_reaction.load("generic.vert", "compute_reaction.frag");

}

void ReactionDiffusion::update(
        int mapFactor, int mapWidth, int mapHeight, 
        float time, float deltaTime, float timeOfDay, float timeOfMonth, 
        int cvDownScale,
        ofFbo flowFbo, ofTexture opticalFlowTexture
    ) 
{

    if (abs(timeOfMonth) < 0.01) {
		int newReactionMap = int(ofRandom(0., 5.));
		if (newReactionMap == 0) {
			newFeedMin = 0.01;
			newFeedRange = 0.09;
			bReSpawnReaction = 1;
		} else if (newReactionMap == 1) {
			newFeedMin = 0.01;
			newFeedRange = 0.025;
			bReSpawnReaction = 1;
		} else if (newReactionMap == 2) {
			newFeedMin = 0.035;
			newFeedRange = 0.025;
			bReSpawnReaction = 1;
		} else if (newReactionMap == 3) {
			newFeedMin = 0.06;
			newFeedRange = 0.015;
			bReSpawnReaction = 1;
		} else if (newReactionMap == 4) {
			newFeedMin = 0.075;
			newFeedRange = 0.015;
			bReSpawnReaction = 1;
		}
		bReSpawnReaction = 0;
	}

    reactionFbo.begin();
	ofClear(255,255,255, 0);
	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", mapWidth, mapHeight);
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_reaction.setUniform1f("reactionFlowMag", reactionFlowMag);
	compute_reaction.setUniform1f("feedMin", feedMin);
	compute_reaction.setUniform1f("feedRange", feedRange);
	compute_reaction.setUniform1f("mapFactor", mapFactor);
	compute_reaction.setUniform1i("initialise", bReSpawnReaction);
	compute_reaction.setUniformTexture("flowMap", flowFbo.getTexture(), 2);
	compute_reaction.setUniformTexture("lastReactionMap", lastReactionFbo.getTexture(), 1);
    compute_reaction.setUniformTexture("optFlowMap", opticalFlowTexture, 3);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_reaction.end();
	reactionFbo.end();

	lastReactionFbo.begin();
	ofClear(255, 255, 255, 0);
	reactionFbo.draw(0, 0, mapWidth, mapHeight);
	lastReactionFbo.end();

	if (bReSpawnReaction == 1) {
		bReSpawnReaction = 0;
	}
}

void ReactionDiffusion::draw() {
	reactionFbo.draw(0, 0);
}

ofFbo ReactionDiffusion::getReactionFbo() {
	return reactionFbo;
}

void ReactionDiffusion::reloadShaders() {
    compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();
}

void ReactionDiffusion::reSpawnReaction() {
	bReSpawnReaction = 1;
}

void ReactionDiffusion::setReactionFeedRange(float feedMin, float feedRange) {
	newFeedMin = feedMin;
	newFeedRange = feedRange;
}

void ReactionDiffusion::setReactionFlowMag(float reactionFlowMag) {
	newReactionFlowMag = reactionFlowMag;
}

void ReactionDiffusion::exit() {
    reactionFbo.clear();
	lastReactionFbo.clear();
}

void ReactionDiffusion::copyVariables() {
	newFeedMin = feedMin;
	newFeedRange = feedRange;

	newReactionFlowMag = reactionFlowMag;
}

void ReactionDiffusion::moveToVariables() {
    float rate = 0.02;

	feedMin = (1 - rate) * feedMin + rate * newFeedMin;
	feedRange = (1 - rate) * feedRange + rate * newFeedRange;

	reactionFlowMag = (1 - rate) * reactionFlowMag + rate * newReactionFlowMag;
}