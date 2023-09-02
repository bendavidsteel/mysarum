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

	feedMin = 0.01;
    feedRange = 0.09;
	bReSpawnReaction = 1;

	reactionFlowMag = 0.;
	agentFlowMag = 0.;

	// slime constants
	diffuseRate = 0.2;
	decayRate = 0.98;
	trailWeight = 1;

	// slime setup
	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	allSpecies[0].movementAttributes.x = 1.1 * mapFactor; // moveSpeed
	allSpecies[0].movementAttributes.y = 0.04 * 2 * PI; // turnSpeed
	allSpecies[0].movementAttributes.z = CIRCLE; //spawn
	allSpecies[0].sensorAttributes.x = 30 * PI / 180; // sensorAngleRad
	allSpecies[0].sensorAttributes.y = 10 * mapFactor; // sensorOffsetDist
	allSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);

	allSpecies[1].movementAttributes.x = 0.9 * mapFactor; // moveSpeed
	allSpecies[1].movementAttributes.y = 0.08 * 2 * PI; // turnSpeed
	allSpecies[1].movementAttributes.z = RING;
	allSpecies[1].sensorAttributes.x = 40 * PI/ 180; // sensorAngleRad
	allSpecies[1].sensorAttributes.y = 20 * mapFactor; //sensorOffsetDist
	allSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);
	newSpecies.resize(numSpecies);

	int numParticles = 1024 * 64;
	particles.resize(numParticles);

	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, mapWidth);
			p.pos.y = ofRandom(0, mapHeight);
		} else if (allSpecies[speciesIdx].movementAttributes.z == CIRCLE) {
			p.pos.x = mapWidth / 2;
			p.pos.y = mapHeight / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * mapWidth;
			p.pos.x = (mapWidth / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (mapHeight / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		}
		p.vel.x = ofRandom(-1, 1);
		p.vel.y = ofRandom(-1, 1);
		p.vel = glm::normalize(p.vel);
		p.vel = p.vel * allSpecies[speciesIdx].movementAttributes.x;
		p.attributes.x = speciesIdx;
	}
	
	particlesBuffer.allocate(particles, GL_DYNAMIC_DRAW);
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	ofPixels initialTrail;
	initialTrail.allocate(mapWidth, mapHeight, OF_PIXELS_RGBA);
	ofColor initialTrailColor(0., 0., 0., 0.);
	initialTrail.setColor(initialTrailColor);

	trailFbo1.allocate(mapWidth, mapHeight, GL_RGBA8);
	trailFbo1.loadData(initialTrail);
	trailFbo2.allocate(mapWidth, mapHeight, GL_RGBA8);
	trailFbo2.loadData(initialTrail);
	particleFbo.allocate(mapWidth, mapHeight, GL_RGBA8);

	// reaction diffusion setup
	reactionFbo.allocate(mapWidth, mapHeight, GL_RG16);
	lastReactionFbo.allocate(mapWidth, mapHeight, GL_RG16);

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

	// general setup
	flowFbo.allocate(mapWidth, mapHeight, GL_RGBA8);

	// load slime shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.load("generic.vert", "compute_decay.frag");
	compute_diffuse.load("generic.vert", "compute_diffuse.frag");

	// load reaction diffusion shaders
	compute_reaction.load("generic.vert", "compute_reaction.frag");

	// load general shaders
	compute_flow.load("generic.vert", "compute_flow.frag");

	renderer.load("generic.vert", "renderer.frag");
	simple_renderer.load("generic.vert", "simple_renderer.frag");
	
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
void Biomass::update(){
	float deltaTime = 1.; //ofGetLastFrameTime();

	float time = ofGetElapsedTimef();
	monthRate = dayRate / 10.;
	float days = time / dayRate;
	float months = time / monthRate;
	time_of_day = fmod(days, float(2 * PI)) - PI;
	time_of_month = fmod(months, float(2 * PI)) - PI;

	// evolve physarum
	if (abs(time_of_day) < 0.1) {
		newSpecies[0].movementAttributes.x = min(max(allSpecies[0].movementAttributes.x + mapFactor * ofRandom(-0.01, 0.01), float(mapFactor * 0.6)), float(mapFactor * 1.4)); // moveSpeed
		newSpecies[0].movementAttributes.y = min(max(allSpecies[0].movementAttributes.y + ofRandom(-0.001, 0.001), float(0.03)), float(0.13)) * 2 * PI; // turnSpeed
		newSpecies[0].sensorAttributes.x = min(max(allSpecies[0].sensorAttributes.x + ofRandom(-0.5, 0.5), float(30.)), float(70.)) * PI / 180; // sensorAngleRad
		newSpecies[0].sensorAttributes.y = min(max(allSpecies[0].sensorAttributes.y + ofRandom(-0.5, 0.5), float(10.)), float(50.)); // sensorOffsetDist
		
		newSpecies[1].movementAttributes.x = min(max(allSpecies[1].movementAttributes.x + mapFactor * ofRandom(-0.01, 0.01), float(mapFactor * 0.6)), float(mapFactor * 1.4)); // moveSpeed
		newSpecies[1].movementAttributes.y = min(max(allSpecies[1].movementAttributes.y + ofRandom(-0.001, 0.001), float(0.03)), float(0.13)) * 2 * PI; // turnSpeed
		newSpecies[1].sensorAttributes.x = min(max(allSpecies[1].sensorAttributes.x + ofRandom(-0.5, 0.5), float(30.)), float(70.)) * PI / 180; // sensorAngleRad
		newSpecies[1].sensorAttributes.y = min(max(allSpecies[1].sensorAttributes.y + ofRandom(-0.5, 0.5), float(10.)), float(50.)); // sensorOffsetDist

		newDiffuseRate = min(max(diffuseRate + ofRandom(-0.01, 0.01), float(0.05)), float(0.4));
		newDecayRate = min(max(decayRate + ofRandom(-0.001, 0.001), float(0.9)), float(0.999));
	}

	if (abs(time_of_month) < 0.01) {
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
		
		int newSlimeColour = int(ofRandom(0., 1.5));
		if (newSlimeColour == 0) {
			newSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);
			newSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);
		} else if (newSlimeColour == 1) {
			newSpecies[0].colour = glm::vec4(0.263, 0.31, 0.98, 1.);
			newSpecies[1].colour = glm::vec4(0.396, 0.839, 0.749, 1.);
		}

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

	vector<Component> thesePoints(points.size());
	float a = 1;//pow(melBands[0], 3) / 2;
	for (int i = 0; i < points.size(); i++) {
		thesePoints[i].value.x = a * points[i].value.x * cos((2 * PI * points[i].value.y) + time_of_day);
		thesePoints[i].value.y = a * points[i].value.x * sin((2 * PI * points[i].value.y) + time_of_day);
		thesePoints[i].value.z = points[i].value.z;
		thesePoints[i].value.w = points[i].value.w;
	}
	pointsBuffer.updateData(thesePoints);

	// handle inputs
	moveToVariables();

	if (bReSpawnAgents) {
		reSpawnAgents();
		bReSpawnAgents = false;
	}

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
	compute_audio.setUniform1f("angle", time_of_day);
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

	// slime updates
	compute_agents.begin();
	compute_agents.setUniform2i("resolution", mapWidth, mapHeight);
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", time);
	compute_agents.setUniform1f("trailWeight", trailWeight);
	compute_agents.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_agents.setUniform1f("agentFlowMag", agentFlowMag);
	compute_agents.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
	compute_agents.setUniformTexture("reactionMap", reactionFbo.getTexture(), 1);
	
	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of particles be < 1024
	compute_agents.dispatchCompute((particles.size() + 1024 -1 )/1024, 1, 1);
	compute_agents.end();

	particleFbo1.begin();
	particleVbo.draw();
	particleFbo1.end();

	trailFbo2.begin();
	compute_decay.begin();
	compute_decay.setUniform2i("resolution", mapWidth, mapHeight);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_decay.setUniformTexture("particleMap", particleFbo.getTexture(), 0);
	compute_decay.setUniformTexture("trailMap", trailFbo1.getTexture(), 1);
	compute_decay.end();
	trailFbo2.end();

	// horizontal blur
	trailFbo1.begin();
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", mapWidth, mapHeight);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2i("blurDir", 1, 0);
	compute_diffuse.setUniformTexture("trailMap", trailFbo2.getTexture(), 1);
	compute_diffuse.end();
	trailFbo1.end();

	// vertical blur
	trailFbo2.begin();
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", mapWidth, mapHeight);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2i("blurDir", 0, 1);
	compute_diffuse.setUniformTexture("trailMap", trailFbo1.getTexture(), 1);
	compute_diffuse.end();
	trailFbo2.end();

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

//--------------------------------------------------------------
void Biomass::draw() {
	float time = ofGetElapsedTimef();
	
	float sun_x = (mapWidth / 2) + (2 * mapWidth / 3) * cos(time_of_day);
	float sun_y = (mapHeight / 2) + (2 * mapHeight / 3) * sin(time_of_day);
	float sun_z = 25 + (10 * sin(time_of_month));

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
		renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
		renderer.setUniformTexture("reactionMap", reactionFbo.getTexture(), 1);
		renderer.setUniformTexture("trailMap", trailFbo2.getTexture(), 2);
		ofSetColor(255);
		ofDrawRectangle(0, 0, mapWidth, mapHeight);
		renderer.end();

	}  else if (display == 1 || display == 2 || display == 3 || display == 4){
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
		simple_renderer.setUniformTexture("trailMap", trailFbo2.getTexture(), 2);
		ofSetColor(255);
		ofDrawRectangle(0, 0, mapWidth, mapHeight);
		simple_renderer.end();
	} else if (display == 5) {
		flowFbo.draw(0, 0, mapWidth, mapHeight);
	}
}

void Biomass::exit(){
	trailMap.clear();
	reactionFbo.clear();
	lastReactionFbo.clear();
	flowFbo.clear();
}

void Biomass::copyVariables() {
	for (int i = 0; i < points.size(); i++) {
		newPoints[i].value.x = points[i].value.x;
		newPoints[i].value.y = points[i].value.y;
		newPoints[i].value.z = points[i].value.z;
		newPoints[i].value.w = points[i].value.w;
	}

	for (int i = 0; i < 2; i++) {
		newSpecies[i].movementAttributes.x = allSpecies[i].movementAttributes.x;
		newSpecies[i].movementAttributes.y = allSpecies[i].movementAttributes.y;
		newSpecies[i].movementAttributes.z = allSpecies[i].movementAttributes.z;
		newSpecies[i].sensorAttributes.x = allSpecies[i].sensorAttributes.x;
		newSpecies[i].sensorAttributes.y = allSpecies[i].sensorAttributes.y;
		newSpecies[i].colour = allSpecies[i].colour;
	}

	newColourA = colourA;
	newColourB = colourB;

	newDayRate = dayRate;
	newChemHeight = chemHeight;
	newTrailHeight = trailHeight;

	newDiffuseRate = diffuseRate;
	newDecayRate = decayRate;
	newTrailWeight = trailWeight;

	newFeedMin = feedMin;
	newFeedRange = feedRange;

	newReactionFlowMag = reactionFlowMag;
	newAgentFlowMag = agentFlowMag;
}

void Biomass::moveToVariables() {
	float rate = 0.02;

	for (int i = 0; i < points.size(); i++) {
		points[i].value = glm::mix(points[i].value, newPoints[i].value, rate);
	}

	for (int i = 0; i < 2; i++) {
		allSpecies[i].movementAttributes.x = (1 - rate) * allSpecies[i].movementAttributes.x + rate * newSpecies[i].movementAttributes.x;
		allSpecies[i].movementAttributes.y = (1 - rate) * allSpecies[i].movementAttributes.y + rate * newSpecies[i].movementAttributes.y;
		allSpecies[i].sensorAttributes = glm::mix(allSpecies[i].sensorAttributes, newSpecies[i].sensorAttributes, rate);
		allSpecies[i].colour = glm::mix(allSpecies[i].colour, newSpecies[i].colour, rate);
	}
	allSpeciesBuffer.updateData(allSpecies);

	colourA = glm::mix(colourA, newColourA, rate);
	colourB = glm::mix(colourB, newColourB, rate);

	chemHeight = (1 - rate) * chemHeight + rate * newChemHeight;
	trailHeight = (1 - rate) * trailHeight + rate * newTrailHeight;

	diffuseRate = (1 - rate) * diffuseRate + rate * newDiffuseRate;
	decayRate = (1 - rate) * decayRate + rate * newDecayRate;
	trailWeight = (1 - rate) * trailWeight + rate * newTrailWeight;

	feedMin = (1 - rate) * feedMin + rate * newFeedMin;
	feedRange = (1 - rate) * feedRange + rate * newFeedRange;

	reactionFlowMag = (1 - rate) * reactionFlowMag + rate * newReactionFlowMag;
	agentFlowMag = (1 - rate) * agentFlowMag + rate * newAgentFlowMag;

	dayRate = (1 - rate) * dayRate + rate * newDayRate;
}

void Biomass::setDayRate(float dayRate) {
	newDayRate = dayRate;
}

void Biomass::reSpawnAgents() {
	allSpecies[0].movementAttributes.z = std::round(ofRandom(0, 5.1));
	allSpecies[1].movementAttributes.z = std::round(ofRandom(0, 5.1));
	int numSpecies = 2;
	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, mapWidth);
			p.pos.y = ofRandom(0, mapHeight);
		} else if (allSpecies[speciesIdx].movementAttributes.z == CIRCLE) {
			p.pos.x = mapWidth / 2;
			p.pos.y = mapHeight / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * mapWidth;
			p.pos.x = (mapWidth / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (mapHeight / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		} else if (allSpecies[speciesIdx].movementAttributes.z == SMALL_RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.15 * mapWidth;
			p.pos.x = (mapWidth / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (mapHeight / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		} else if (allSpecies[speciesIdx].movementAttributes.z == VERTICAL_LINE) {
			p.pos.x = (mapWidth / 2) * ofRandom(0.99, 1.01);
			p.pos.y = ofRandom(0, mapHeight);
		} else if (allSpecies[speciesIdx].movementAttributes.z == HORIZONTAL_LINE) {
			p.pos.x = ofRandom(0, mapWidth);
			p.pos.y = (mapHeight / 2) * ofRandom(0.99, 1.01);
		}
		p.vel.x = ofRandom(-1, 1);
		p.vel.y = ofRandom(-1, 1);
		p.vel = glm::normalize(p.vel);
		p.vel = p.vel * allSpecies[speciesIdx].movementAttributes.x;
		p.attributes.x = speciesIdx;
	}
	particlesBuffer.updateData(particles);
}

void Biomass::setSpeciesColour(glm::vec3 colour1, glm::vec3 colour2) {
	newSpecies[0].colour.x = colour1.x;
	newSpecies[0].colour.y = colour1.y;
	newSpecies[0].colour.z = colour1.z;

	newSpecies[1].colour.x = colour2.x;
	newSpecies[1].colour.y = colour2.y;
	newSpecies[1].colour.z = colour2.z;
}

void Biomass::setAgentFlowMag(float agentFlowMag) {
	newAgentFlowMag = agentFlowMag;
}

void Biomass::reloadShaders() {
	renderer.load("generic.vert", "renderer.frag");
		
	compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();
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

void Biomass::reSpawnReaction() {
	bReSpawnReaction = 1;
}

void Biomass::setReactionFeedRange(float feedMin, float feedRange) {
	newFeedMin = feedMin;
	newFeedRange = feedRange;
}

void Biomass::setReactionColour(glm::vec3 colourA, glm::vec3 colourB) {
	newColourA = colourA;
	newColourB = colourB;
}

void Biomass::setReactionFlowMag(float reactionFlowMag) {
	newReactionFlowMag = reactionFlowMag;
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
