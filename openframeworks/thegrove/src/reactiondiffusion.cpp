#include "reactiondiffusion.h"

void ReactionDiffusion::setup(int _width, int _depth) {
	mapWidth = _width;
	mapHeight = _depth;
	
    feedMin = 0.01;
    feedRange = 0.09;
	bReSpawnReaction = 1;

    reactionFlowMag = 0.;

    // reaction diffusion setup
	reactionFbo.allocate(mapWidth, mapHeight, GL_RGBA16);
	lastReactionFbo.allocate(mapWidth, mapHeight, GL_RGBA16);

	colourA = glm::vec3(1.0, 0.0, 0.0);
	colourB = glm::vec3(0.0, 1.0, 1.0);

    // load reaction diffusion shaders
	reloadShaders();

	bReSpawnReaction = 1;

	plane.set(mapWidth, mapHeight, mapWidth, mapHeight);
	plane.setPosition(mapWidth / 2., 0., mapHeight / 2.);
	plane.rotateRad(PI / 2., 1., 0., 0.);
	plane.mapTexCoordsFromTexture(reactionFbo.getTexture());
}

void ReactionDiffusion::update() 
{
	float deltaTime = 1.;

	feedMin = 0.01;
	feedRange = 0.09;

    reactionFbo.begin();
	ofClear(255,255,255, 255);
	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", mapWidth, mapHeight);
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.setUniform1f("feedMin", feedMin);
	compute_reaction.setUniform1f("feedRange", feedRange);
	compute_reaction.setUniform1i("initialise", bReSpawnReaction);
	compute_reaction.setUniformTexture("lastReactionMap", lastReactionFbo.getTexture(), 0);
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
	reactionFbo.draw(0, 0, mapWidth, mapHeight);
	// plane_renderer.begin();
	// plane_renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
	// plane_renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
	// plane_renderer.setUniformTexture("reactionMap", reactionFbo.getTexture(), 0);
	
	// ofPushMatrix();

	// // translate plane into center screen.
	// // ofTranslate(mapWidth / 2, mapHeight / 2, 0);
	// // ofRotateDeg(90., 1, 0, 0);

	// plane.draw();

	// ofPopMatrix();

	// plane_renderer.end();
}

ofFbo ReactionDiffusion::getReactionFbo() {
	return reactionFbo;
}

void ReactionDiffusion::reloadShaders() {
    compute_reaction.load("generic.vert", "compute_reaction.frag");
	plane_renderer.load("plane_renderer.vert", "plane_renderer.frag");
}

void ReactionDiffusion::exit() {
    reactionFbo.clear();
	lastReactionFbo.clear();
}