#pragma once
#include "ofMain.h"
#include "agent.h"
#include "utils.h"

class Species{
	public:
		Species();

		float moveSpeed;
		float turnSpeed;
		float sensorAngleDegrees;
		float sensorOffsetDist;
		int sensorSize;
		valarray<float> colour;
};

class Slime{

	public:
		Slime();
		void setup(int numAgents);
		void update();
		void draw(ofPixels& pixels);

		void diffuse();
		float sense(Agent agent, Species species, float sensorAngleOffset);

		vector<Species> allSpecies;
		int numSpecies;

		vector<Agent> agents;
		int numAgents;

		static const int mapHeight = 1024;
		static const int mapWidth = 768;

		std::array<valarray<float>, mapHeight * mapWidth> trailMap;

		float deltaTime;
		float trailWeight;
		float decayRate;
		float diffuseRate;
};
