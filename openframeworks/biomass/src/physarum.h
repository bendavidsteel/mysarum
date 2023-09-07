#pragma once

#include "ofMain.h"

enum Spawn{
	RANDOM=0,
	CIRCLE=1,
	RING=2,
	SMALL_RING=3,
	VERTICAL_LINE=4,
	HORIZONTAL_LINE=5
};

class Physarum {
    public:
        void setup(int mapWidth, int mapHeight, int mapFactor);
        void update(
            int mapFactor, int mapWidth, int mapHeight, 
            float time, float deltaTime, float timeOfDay, float timeOfMonth, 
            int cvDownScale,
            ofFbo flowFbo, ofFbo reactionFbo, ofFbo audioFbo, ofTexture opticalFlowTexture
        );
        void draw(int mapWidth, int mapHeight);

		ofFbo getTrailFbo();

        void reSpawnAgents(int mapWidth, int mapHeight);
		void setSpeciesColour(glm::vec3 colour1, glm::vec3 colour2);
		void setAgentFlowMag(float agentFlowMag);

        void reloadShaders();

        void exit();

        void copyVariables();
        void moveToVariables();

    private:
        // slime
		struct Agent{
			glm::vec2 pos;
			glm::vec2 vel;
			glm::vec4 speciesMask;
		};

		struct Species{
			glm::vec4 colour;
			glm::vec4 movementAttributes;
			glm::vec4 sensorAttributes;
		};

		ofShader compute_agents;
		ofShader compute_decay;
		ofShader compute_diffuse;

		vector<Agent> agents;
		ofBufferObject agentBuffer;
		vector<Species> allSpecies;
		vector<Species> newSpecies;
		ofBufferObject allSpeciesBuffer;

		ofFbo trailFbo1;
		ofFbo trailFbo2;
		ofFbo agentFbo;

        float diffuseRate;
		float newDiffuseRate;
		float decayRate;
		float newDecayRate;
		float trailWeight;
		float newTrailWeight;

        float agentFlowMag;
		float newAgentFlowMag;

        bool bReSpawnAgents;
};