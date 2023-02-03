#include "slime.h"

Species::Species(float sensorAngleDegrees, float turnSpeedMag) {
	sensorAngleRad = sensorAngleDegrees * PI / 180;
	turnSpeed = turnSpeedMag * 2 * PI;
}

//------------------------------------------------------------------
Slime::Slime() {
}

void Slime::setup(int numAgents)
{
	this->numAgents = numAgents;
	trailWeight = 5;
	decayRate = 1.0;
	diffuseRate = 0.0;
	deltaTime = 1.0;

	numSpecies = 1;
	allSpecies = vector<Species>(numSpecies, Species(45, 0.1));
	// species 1
	allSpecies[0].colour = {255, 255, 255};
	allSpecies[0].moveSpeed = 1.0;
	allSpecies[0].sensorOffsetDist = 15;
	allSpecies[0].sensorSize = 1;

	int species = 0;
	AgentSpawn spawnStrategy = CENTRE;
	agents = vector<Agent>(numAgents, Agent(species, numSpecies, spawnStrategy));

	for (int i = 0; i < mapHeight * mapWidth; i++) {
		trailMap[i] = valarray<float>(0.0, numSpecies);
	}
}

//------------------------------------------------------------------
void Slime::update(){
	for (int i = 0; i < numAgents; i++)
	{
		Agent & agent = agents[i];
		Species species = allSpecies[agent.speciesIndex];

		// Steer based on sensory data
		float sensorAngleRad = species.sensorAngleRad;
		float weightForward = sense(agent, species, 0);
		float weightLeft = sense(agent, species, sensorAngleRad);
		float weightRight = sense(agent, species, -sensorAngleRad);

		
		float randomSteerStrength = rand() / RAND_MAX;
		float turnSpeed = species.turnSpeed * 2 * PI;

		// Continue in same direction
		if ((weightForward > weightLeft) && (weightForward > weightRight)) {
			agent.angle += 0;
		}
		else if (weightForward < weightLeft && weightForward < weightRight) { // TODO does this make sense?
			agent.angle += (randomSteerStrength - 0.5) * 2 * turnSpeed * deltaTime;
		}
		// Turn right
		else if (weightRight > weightLeft) {
			agent.angle -= randomSteerStrength * turnSpeed * deltaTime;
		}
		// Turn left
		else if (weightLeft > weightRight) {
			agent.angle += randomSteerStrength * turnSpeed * deltaTime;
		}

		// Update position
		agent.updatePosition(deltaTime, species.moveSpeed);
		
		// Clamp position to map boundaries, and pick new random move dir if hit boundary
		agent.ensureRebound();

		int x = int(agent.position.x);
		int y = int(agent.position.y);
		valarray<float> oldTrail = trailMap[(y * ofGetWidth()) + x];
		valarray<float> newTrail = min((oldTrail + (agent.speciesMask * trailWeight * deltaTime)), 1.0);
		trailMap[(y * ofGetWidth()) + x] = newTrail;
	} //<>//
	diffuse();
}

//------------------------------------------------------------------
void Slime::draw(ofPixels& pixels) {

	int x;
	int y;
	for(auto line: pixels.getLines())
	{
		x = 0;
		y = line.getLineNum();
		for(auto pixel: line.getPixels())
		{
			valarray<float> map = trailMap[(y * ofGetWidth()) + x];
			
			valarray<float> colour(0.0, 3);
			for (int i = 0; i < numSpecies; i ++) {
				valarray<float> mask = toOneHot(i, numSpecies);
				colour += allSpecies[i].colour * dot(map, mask); 
			}
			
			pixel[0] = colour[0];
			pixel[1] = colour[1];
			pixel[2] = colour[2];
			x++;
		}
	}
}

float Slime::sense(Agent agent, Species species, float sensorAngleOffset) {
	float sensorAngle = agent.angle + sensorAngleOffset;
	glm::vec2 sensorDir = glm::vec2(cos(sensorAngle), sin(sensorAngle));

	glm::vec2 sensorPos = agent.position + (sensorDir * species.sensorOffsetDist);
	int sensorCentreX = (int) sensorPos.x;
	int sensorCentreY = (int) sensorPos.y;

	float sum = 0;

	valarray<float> senseWeight = (agent.speciesMask * 2.0) - 1.0;
	for (int offsetX = -species.sensorSize; offsetX <= species.sensorSize; offsetX ++) {
		for (int offsetY = -species.sensorSize; offsetY <= species.sensorSize; offsetY ++) {
		int sampleX = min(ofGetWidth() - 1, max(0, sensorCentreX + offsetX));
		int sampleY = min(ofGetHeight() - 1, max(0, sensorCentreY + offsetY));
		sum += dot(senseWeight, trailMap[(sampleY * ofGetWidth()) + sampleX]);
		}
	}

	return sum;
}

void Slime::diffuse() {
	for (int y = 0; y < ofGetHeight(); y++)
	{
		for (int x = 0; x < ofGetWidth(); x++)
		{
			valarray<float> sum(0.0, numSpecies);
			valarray<float> originalCol = trailMap[(y * ofGetWidth()) + x];
			// 3x3 blur
			for (int offsetX = -1; offsetX <= 1; offsetX ++) {
				for (int offsetY = -1; offsetY <= 1; offsetY ++) {
				int sampleX = min(ofGetWidth()-1, max(0, x + offsetX));
				int sampleY = min(ofGetHeight()-1, max(0, y + offsetY));
				sum += trailMap[(sampleY * ofGetWidth()) + sampleX];
				}
			}
			
			valarray<float> blurredCol = sum / 9;
			float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
			blurredCol = (originalCol * (1 - diffuseWeight)) + (blurredCol * diffuseWeight);
			
			//DiffusedTrailMap[id.xy] = blurredCol * saturate(1 - decayRate * deltaTime);
			trailMap[(y * ofGetWidth()) + x] = max((blurredCol * decayRate * deltaTime), 0.0);
		}
	}
}
