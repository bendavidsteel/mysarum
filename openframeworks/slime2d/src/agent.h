#pragma once
#include "ofMain.h"
#include "utils.h"

enum AgentSpawn{
	RANDOM,
	CENTRE
};

class Agent{

	public:
		Agent(int species, int numSpecies, AgentSpawn spawnStrategy);

		void spawn(AgentSpawn spawnStrategy);
		void ensureRebound();
		void doRebound(float randomAngle);
		glm::vec2 getVelocity();
		void updatePosition(float deltaTime, float moveSpeed);
		
		glm::vec2 position;
		glm::float32 angle;
		valarray<glm::float32> speciesMask;
		glm::int8 speciesIndex;

		int boidColour;
		float boidSize;
};
