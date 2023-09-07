#include "physarum.h"

void Physarum::setup(int mapWidth, int mapHeight, int mapFactor) {
    // slime constants
	diffuseRate = 1.;
	decayRate = 0.98;
	trailWeight = 1;

    agentFlowMag = 0.;

	// slime setup
	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	allSpecies[0].movementAttributes.x = 1.1 * mapFactor; // moveSpeed
	allSpecies[0].movementAttributes.y = 0.04 * 2 * PI; // turnSpeed
	allSpecies[0].movementAttributes.z = RANDOM; //spawn
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

	int numAgents = 1024 * 64;
	agents.resize(numAgents);

	int speciesIdx = 0;
	for(int idx = 0; idx < agents.size(); idx++){
		auto &p = agents[idx];
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
		
		// convert speciesIdx to speciesMask glm::vec4
		p.speciesMask = glm::vec4(0., 0., 0., 0.);
		if (speciesIdx == 0) {
			p.speciesMask.x = 1;
		} else if (speciesIdx == 1) {
			p.speciesMask.y = 1;
		} else if (speciesIdx == 2) {
			p.speciesMask.z = 1;
		} else if (speciesIdx == 3) {
			p.speciesMask.w = 1;
		}
	}
	
	agentBuffer.allocate(agents, GL_DYNAMIC_DRAW);
	agentBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	ofPixels initialTrail;
	initialTrail.allocate(mapWidth, mapHeight, OF_PIXELS_RGBA);
	ofColor initialTrailColor(0., 0., 0., 0.);
	initialTrail.setColor(initialTrailColor);

	trailFbo1.allocate(mapWidth, mapHeight, GL_RGBA8);
	trailFbo1.begin();
	ofClear(255,255,255, 0);
	trailFbo1.end();
	trailFbo2.allocate(mapWidth, mapHeight, GL_RGBA8);
	trailFbo2.begin();
	ofClear(255,255,255, 0);
	trailFbo2.end();
	
    agentFbo.allocate(mapWidth, mapHeight, GL_RGBA8);
    agentFbo.getTexture().bindAsImage(3, GL_READ_WRITE);

    // load slime shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.load("generic.vert", "compute_decay.frag");
	compute_diffuse.load("generic.vert", "compute_diffuse.frag");

}

void Physarum::update(
        int mapFactor, int mapWidth, int mapHeight, 
        float time, float deltaTime, float timeOfDay, float timeOfMonth, 
        int cvDownScale,
        ofFbo flowFbo, ofFbo reactionFbo, ofFbo audioFbo, ofTexture optFlowTexture
    )
{
    if (bReSpawnAgents) {
		reSpawnAgents(mapWidth, mapHeight);
		bReSpawnAgents = false;
	}

    // evolve physarum
	if (abs(timeOfDay) < 0.1) {
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

    if (abs(timeOfMonth) < 0.01) {
		int newSlimeColour = int(ofRandom(0., 1.5));
		if (newSlimeColour == 0) {
			newSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);
			newSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);
		} else if (newSlimeColour == 1) {
			newSpecies[0].colour = glm::vec4(0.263, 0.31, 0.98, 1.);
			newSpecies[1].colour = glm::vec4(0.396, 0.839, 0.749, 1.);
		}
	}

	// clear agent fbo each time so that it only tracks agents current state
	agentFbo.begin();
	ofClear(0, 0, 0, 0);
	agentFbo.end();

    // slime updates
	compute_agents.begin();
	compute_agents.setUniform2i("resolution", mapWidth, mapHeight);
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", time);
	compute_agents.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_agents.setUniform1f("agentFlowMag", agentFlowMag);
	compute_agents.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
	compute_agents.setUniformTexture("reactionMap", reactionFbo.getTexture(), 1);
	compute_agents.setUniformTexture("trailMap", trailFbo2.getTexture(), 2);
	compute_agents.setUniformTexture("audioMap", audioFbo.getTexture(), 3);
	compute_agents.setUniformTexture("optFlowMap", optFlowTexture, 4);
	
	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of agents be < 1024
	compute_agents.dispatchCompute((agents.size() + 1024 -1 )/1024, 1, 1);
	compute_agents.end();

    // we could use a vbo here, but it's going to be annoying to use an fbo to capture 
    // vbo points that are outside the normal window size?

	// TODO consider using more fbos to avoid weird screen stuff?

	trailFbo2.begin();
	ofClear(255,255,255, 0);
	compute_decay.begin();
	compute_decay.setUniform2i("resolution", mapWidth, mapHeight);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.setUniform1f("trailWeight", trailWeight);
	compute_decay.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_decay.setUniformTexture("agentMap", agentFbo.getTexture(), 0);
	compute_decay.setUniformTexture("trailMap", trailFbo1.getTexture(), 1);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_decay.end();
	trailFbo2.end();

	// horizontal blur
	trailFbo1.begin();
	ofClear(255,255,255, 0);
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", mapWidth, mapHeight);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2f("blurDir", 1., 0.);
	compute_diffuse.setUniformTexture("trailMap", trailFbo2.getTexture(), 1);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_diffuse.end();
	trailFbo1.end();

	// vertical blur
	trailFbo2.begin();
	ofClear(255,255,255, 0);
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", mapWidth, mapHeight);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2f("blurDir", 0., 1.);
	compute_diffuse.setUniformTexture("trailMap", trailFbo1.getTexture(), 1);
	ofSetColor(255);
	ofDrawRectangle(0, 0, mapWidth, mapHeight);
	compute_diffuse.end();
	trailFbo2.end();

	// // draw fbo2 into fbo1 ready for the next update loop
	trailFbo1.begin();
	ofClear(255,255,255, 0);
	trailFbo2.draw(0, 0, mapWidth, mapHeight);
	trailFbo1.end();
}

void Physarum::draw(int mapWidth, int mapHeight) {
	trailFbo2.draw(0, 0, mapWidth, mapHeight);
}

ofFbo Physarum::getTrailFbo() {
	return trailFbo2;
}

void Physarum::reSpawnAgents(int mapWidth, int mapHeight) {
	allSpecies[0].movementAttributes.z = std::round(ofRandom(0, 5.1));
	allSpecies[1].movementAttributes.z = std::round(ofRandom(0, 5.1));
	int numSpecies = 2;
	int speciesIdx = 0;
	for(int idx = 0; idx < agents.size(); idx++){
		auto &p = agents[idx];
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
		
		p.speciesMask = glm::vec4(0., 0., 0., 0.);
		if (speciesIdx == 0) {
			p.speciesMask.x = 1;
		} else if (speciesIdx == 1) {
			p.speciesMask.y = 1;
		} else if (speciesIdx == 2) {
			p.speciesMask.z = 1;
		} else if (speciesIdx == 3) {
			p.speciesMask.w = 1;
		}
	}
	agentBuffer.updateData(agents);
}

void Physarum::setSpeciesColour(glm::vec3 colour1, glm::vec3 colour2) {
	newSpecies[0].colour.x = colour1.x;
	newSpecies[0].colour.y = colour1.y;
	newSpecies[0].colour.z = colour1.z;

	newSpecies[1].colour.x = colour2.x;
	newSpecies[1].colour.y = colour2.y;
	newSpecies[1].colour.z = colour2.z;
}

void Physarum::setAgentFlowMag(float agentFlowMag) {
	newAgentFlowMag = agentFlowMag;
}

void Physarum::reloadShaders() {
    compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
    compute_agents.linkProgram();
}

void Physarum::exit() {
    trailFbo1.clear();
	trailFbo2.clear();
	agentFbo.clear();
}

void Physarum::copyVariables() {
	for (int i = 0; i < 2; i++) {
		newSpecies[i].movementAttributes.x = allSpecies[i].movementAttributes.x;
		newSpecies[i].movementAttributes.y = allSpecies[i].movementAttributes.y;
		newSpecies[i].movementAttributes.z = allSpecies[i].movementAttributes.z;
		newSpecies[i].sensorAttributes.x = allSpecies[i].sensorAttributes.x;
		newSpecies[i].sensorAttributes.y = allSpecies[i].sensorAttributes.y;
		newSpecies[i].colour = allSpecies[i].colour;
	}

	newDiffuseRate = diffuseRate;
	newDecayRate = decayRate;
	newTrailWeight = trailWeight;

	newAgentFlowMag = agentFlowMag;
}

void Physarum::moveToVariables() {
	float rate = 0.02;

	for (int i = 0; i < 2; i++) {
		allSpecies[i].movementAttributes.x = (1 - rate) * allSpecies[i].movementAttributes.x + rate * newSpecies[i].movementAttributes.x;
		allSpecies[i].movementAttributes.y = (1 - rate) * allSpecies[i].movementAttributes.y + rate * newSpecies[i].movementAttributes.y;
		allSpecies[i].sensorAttributes = glm::mix(allSpecies[i].sensorAttributes, newSpecies[i].sensorAttributes, rate);
		allSpecies[i].colour = glm::mix(allSpecies[i].colour, newSpecies[i].colour, rate);
	}
	allSpeciesBuffer.updateData(allSpecies);

	diffuseRate = (1 - rate) * diffuseRate + rate * newDiffuseRate;
	decayRate = (1 - rate) * decayRate + rate * newDecayRate;
	trailWeight = (1 - rate) * trailWeight + rate * newTrailWeight;

	agentFlowMag = (1 - rate) * agentFlowMag + rate * newAgentFlowMag;
}