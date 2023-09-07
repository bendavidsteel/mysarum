#include "biomass.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void Biomass::setup(){

	ofSetFrameRate(120);

	// constants
	display = 0;

	mapFactor = 2.;
	mapWidth = ofGetWidth() * mapFactor;
	mapHeight = ofGetHeight() * mapFactor;

	colourA = glm::vec3(1., 0., 0.);
	colourB = glm::vec3(0., 1., 0.);
	chemHeight = 1.;
	trailHeight = 2.;
	dayRate = 10.;
	monthRate = 10.;

	numPoints = 4;
	points.resize(numPoints);
	for (int i = 0; i < 1; i++) {
		points[i].value.x = 0;
		points[i].value.y = 0;
		points[i].value.z = 1;
		points[i].value.w = 0;
	}
	for (int i = 1; i < 4; i++) {
		points[i].value.x = 0;
		points[i].value.y = 0;
		points[i].value.z = 0;
		points[i].value.w = 0;
	}
	newPoints.resize(numPoints);
	
	pointsBuffer.allocate(points, GL_DYNAMIC_DRAW);
	pointsBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 6);

	compute_audio.load("generic.vert", "compute_audio.frag");

	audioFbo.allocate(mapWidth, mapHeight, GL_RG16);
	trailFbo.allocate(mapWidth, mapHeight, GL_RGBA8);
	reactionFbo.allocate(mapWidth, mapHeight, GL_RGBA8);

	// general setup
	flowFbo.allocate(mapWidth, mapHeight, GL_RGBA8);

	physarum.setup(mapWidth, mapHeight, mapFactor);
	reactionDiffusion.setup(mapWidth, mapHeight);

	// load general shaders
	compute_flow.load("generic.vert", "compute_flow.frag");

	renderer.load("generic.vert", "renderer.frag");
	simple_renderer.load("generic.vert", "simple_renderer.frag");
	physarum_renderer.load("generic.vert", "physarum_renderer.frag");
	reaction_renderer.load("generic.vert", "reaction_renderer.frag");

	int planeWidth = ofGetWidth() * mapFactor;
	int planeHeight = ofGetHeight() * mapFactor;
	int planeGridSize = 1;
	int planeColums = planeWidth / planeGridSize;
	int planeRows = planeHeight / planeGridSize;

	plane.set(planeWidth, planeHeight, planeColums, planeRows, OF_PRIMITIVE_TRIANGLES);
	
	copyVariables();
}

void Biomass::setupAudio(vector<Component> audio) {
	audioBuffer.allocate(audio, GL_DYNAMIC_DRAW);
	audioBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 5);
}

void Biomass::updateAudio(vector<Component> audio) {
	audioBuffer.updateData(audio);
}

//--------------------------------------------------------------
void Biomass::update(ofTexture opticalFlowTexture){
	float deltaTime = 1.; //ofGetLastFrameTime();

	float time = ofGetElapsedTimef();
	monthRate = dayRate / 10.;
	float days = time / dayRate;
	float months = time / monthRate;
	timeOfDay = fmod(days, float(2 * PI)) - PI;
	timeOfMonth = fmod(months, float(2 * PI)) - PI;

	if (abs(timeOfMonth) < 0.01) {
		int newReactionColour = int(ofRandom(0., 10.));

		if (newReactionColour == 0) {
			newColourA = glm::vec3(1., 0., 0.);
			newColourB = glm::vec3(0., 1., 0.);
		} else if (newReactionColour == 1) {
			newColourA = glm::vec3(1.0, 0.906, 0.51);
			newColourB = glm::vec3(0.98, 0.345, 0.118);
		} else if (newReactionColour == 2) {
			newColourA = glm::vec3(0.494, 0.921, 0.063);
			newColourB = glm::vec3(0.839, 0.812, 0.153);
		} else if (newReactionColour == 3) {
			newColourA = glm::vec3(0.839, 0.02, 0.004);
			newColourB = glm::vec3(0., 0., 1.);
		} else if (newReactionColour == 4) {
			newColourA = glm::vec3(1., 0., 0.);
			newColourB = glm::vec3(0., 0., 1.);
		} else if (newReactionColour == 5) {
			newColourA = glm::vec3(191./255., 11./255., 59./255.);
			newColourB = glm::vec3(213./255., 13./255., 216./255.);
		}
	}

	vector<Component> thesePoints(points.size());
	float a = 1;//pow(melBands[0], 3) / 2;
	for (int i = 0; i < points.size(); i++) {
		thesePoints[i].value.x = a * points[i].value.x * cos((2 * PI * points[i].value.y) + timeOfDay);
		thesePoints[i].value.y = a * points[i].value.x * sin((2 * PI * points[i].value.y) + timeOfDay);
		thesePoints[i].value.z = points[i].value.z;
		thesePoints[i].value.w = points[i].value.w;
	}
	pointsBuffer.updateData(thesePoints);

	// handle inputs
	moveToVariables();

	int workGroupSize = 20;

	int widthWorkGroups = ceil(mapWidth/workGroupSize);
	int heightWorkGroups = ceil(mapHeight/workGroupSize);

	// general updates
	audioFbo.begin();
	ofClear(255,255,255, 0);
	compute_audio.begin();
	compute_audio.setUniform2i("resolution", mapWidth, mapHeight);
	compute_audio.setUniform1f("deltaTime", deltaTime);
	compute_audio.setUniform1i("numBands", audioArraySize);
	compute_audio.setUniform1f("angle", timeOfDay);
	compute_audio.setUniform1f("rms", 1);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_audio.end();
	audioFbo.end();

	flowFbo.begin();
	ofClear(255,255,255, 0);
	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", mapWidth, mapHeight);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_flow.end();
	flowFbo.end();

	physarum.update(
		mapFactor, mapWidth, mapHeight, 
		time, deltaTime, timeOfDay, timeOfMonth, 
		cvDownScale,
		flowFbo, reactionDiffusion.getReactionFbo(), audioFbo, opticalFlowTexture
	);

	reactionDiffusion.update(
		mapFactor, mapWidth, mapHeight, 
		time, deltaTime, timeOfDay, timeOfMonth, 
		cvDownScale,
		flowFbo, opticalFlowTexture
	);
}

//--------------------------------------------------------------
void Biomass::draw() {
	float time = ofGetElapsedTimef();
	
	float sun_x = (mapWidth / 2) + (2 * mapWidth / 3) * cos(timeOfDay);
	float sun_y = (mapHeight / 2) + (2 * mapHeight / 3) * sin(timeOfDay);
	float sun_z = 25 + (10 * sin(timeOfMonth));

	glm::vec3 colourC = glm::vec3(1., 0., 0.);
	glm::vec3 colourD = glm::vec3(0., 0., 1.);

	reactionFbo.begin();
	reactionDiffusion.draw();
	reactionFbo.end();

	trailFbo.begin();
	// ofClear(255, 255, 255, 0);
	physarum.draw(mapWidth, mapHeight);
	trailFbo.end();

	// trailFbo.draw(0, 0);

	display = 1;
	if (display == 0) {
		ofClear(255,255,255, 0);
		renderer.begin();
		renderer.setUniform2i("resolution", mapWidth, mapHeight);
		renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
		renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
		renderer.setUniform3f("light", sun_x, sun_y, sun_z);
		renderer.setUniform1f("chem_height", chemHeight);
		renderer.setUniform1f("trail_height", trailHeight);
		renderer.setUniform1f("time", time);
		renderer.setUniform1f("bpm", bps);
		renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 4);
		renderer.setUniformTexture("reactionMap", reactionFbo.getTexture(), 5);
		renderer.setUniformTexture("trailMap", trailFbo.getTexture(), 6);
		
		// plane.draw();
		ofSetColor(255);
		ofDrawRectangle(0, 0, mapWidth, mapHeight);
		// ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());

		renderer.end();

	} else if (display == 1) {
		physarum_renderer.begin();
		physarum_renderer.setUniform2i("resolution", mapWidth, mapHeight);
		physarum_renderer.setUniform3f("colourC", colourC.x, colourC.y, colourC.z);
		physarum_renderer.setUniform3f("colourD", colourD.x, colourD.y, colourD.z);
		physarum_renderer.setUniform3f("light", sun_x, sun_y, sun_z);
		physarum_renderer.setUniform1f("trail_height", trailHeight);
		physarum_renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 1);
		physarum_renderer.setUniformTexture("trailMap", trailFbo.getTexture(), 4);
		physarum_renderer.setUniformTexture("audioMap", audioFbo.getTexture(), 2);

		ofSetColor(255);
		ofDrawRectangle(0, 0, mapWidth, mapHeight);

		physarum_renderer.end();

	} else if (display == 2) {
		reaction_renderer.begin();
		reaction_renderer.setUniform2i("resolution", mapWidth, mapHeight);
		reaction_renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
		reaction_renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
		reaction_renderer.setUniform3f("colourC", colourC.x, colourC.y, colourC.z);
		reaction_renderer.setUniform3f("colourD", colourD.x, colourD.y, colourD.z);
		reaction_renderer.setUniform3f("light", sun_x, sun_y, sun_z);
		reaction_renderer.setUniform1f("chem_height", chemHeight);
		reaction_renderer.setUniformTexture("reactionMap", reactionFbo.getTexture(), 0);
		reaction_renderer.setUniformTexture("audioMap", audioFbo.getTexture(), 1);

		ofSetColor(255);
		ofDrawRectangle(0, 0, mapWidth, mapHeight);

		reaction_renderer.end();

	} else if (display == 3 || display == 4) {
		ofClear(255,255,255, 0);
		simple_renderer.begin();
		simple_renderer.setUniform2i("resolution", mapWidth, mapHeight);
		simple_renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
		simple_renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
		simple_renderer.setUniform3f("light", sun_x, sun_y, sun_z);
		simple_renderer.setUniform1f("chem_height", chemHeight);
		simple_renderer.setUniform1f("trail_height", trailHeight);
		simple_renderer.setUniform1i("opticalFlowDownScale", cvDownScale);
		simple_renderer.setUniform1i("display", display);
		simple_renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
		simple_renderer.setUniformTexture("reactionMap", reactionFbo.getTexture(), 1);
		simple_renderer.setUniformTexture("trailMap", trailFbo.getTexture(), 2);
		
		ofSetColor(255);
		ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());

		simple_renderer.end();
	} else if (display == 5) {
		flowFbo.draw(0, 0, ofGetWidth(), ofGetHeight());
	}
}

void Biomass::exit(){
	physarum.exit();
	reactionDiffusion.exit();
	flowFbo.clear();
	trailFbo.clear();
	reactionFbo.clear();
}

void Biomass::copyVariables() {
	for (int i = 0; i < points.size(); i++) {
		newPoints[i].value.x = points[i].value.x;
		newPoints[i].value.y = points[i].value.y;
		newPoints[i].value.z = points[i].value.z;
		newPoints[i].value.w = points[i].value.w;
	}

	newColourA = colourA;
	newColourB = colourB;

	newDayRate = dayRate;
	newChemHeight = chemHeight;
	newTrailHeight = trailHeight;

	physarum.copyVariables();
	reactionDiffusion.copyVariables();
}

void Biomass::moveToVariables() {
	float rate = 0.02;

	for (int i = 0; i < points.size(); i++) {
		points[i].value = glm::mix(points[i].value, newPoints[i].value, rate);
	}

	colourA = glm::mix(colourA, newColourA, rate);
	colourB = glm::mix(colourB, newColourB, rate);

	chemHeight = (1 - rate) * chemHeight + rate * newChemHeight;
	trailHeight = (1 - rate) * trailHeight + rate * newTrailHeight;

	dayRate = (1 - rate) * dayRate + rate * newDayRate;

	physarum.moveToVariables();
	reactionDiffusion.moveToVariables();
}

void Biomass::setDayRate(float dayRate) {
	newDayRate = dayRate;
}

void Biomass::reloadShaders() {
	renderer.load("generic.vert", "renderer.frag");
	reactionDiffusion.reloadShaders();
}

void Biomass::setDisplay(int d) {
	display = d;
}

void Biomass::setPoints(vector<Component> points) {
	newPoints = points;
}

int Biomass::getNumPoints() {
	return numPoints;
}

int Biomass::getMapWidth() {
	return mapWidth;
}

int Biomass::getMapHeight() {
	return mapHeight;
}

void Biomass::setBPS(float val) {
	bps = val;
}

void Biomass::setCVDownScale(int downScale) {
	cvDownScale = downScale;
}

void Biomass::reSpawnAgents() {
	physarum.reSpawnAgents(mapWidth, mapHeight);
}

void Biomass::setSpeciesColour(glm::vec3 colour1, glm::vec3 colour2) {
	physarum.setSpeciesColour(colour1, colour2);
}

void Biomass::setAgentFlowMag(float agentFlowMag) {
	physarum.setAgentFlowMag(agentFlowMag);
}

void Biomass::reSpawnReaction() {
	reactionDiffusion.reSpawnReaction();
}

void Biomass::setReactionFeedRange(float feedMin, float feedRange) {
	reactionDiffusion.setReactionFeedRange(feedMin, feedRange);
}

void Biomass::setReactionColour(glm::vec3 colourA, glm::vec3 colourB) {
	newColourA = colourA;
	newColourB = colourB;
}

void Biomass::setReactionFlowMag(float reactionFlowMag) {
	reactionDiffusion.setReactionFlowMag(reactionFlowMag);
}
