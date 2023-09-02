#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"

enum Spawn{
	RANDOM=0,
	CIRCLE=1,
	RING=2,
	SMALL_RING=3,
	VERTICAL_LINE=4,
	HORIZONTAL_LINE=5
};

struct Component{
	glm::vec4 value;
};

class Biomass{

	public:
		void setup();
		void update();
		void draw();
		void exit();

		void copyVariables();
		void moveToVariables();

		void setupAudio(vector<Component> audio);
		void updateAudio(vector<Component> audio);

		void setDayRate(float dayRate);

		void reSpawnAgents();
		void setSpeciesColour(glm::vec3 colour1, glm::vec3 colour2);
		void setAgentFlowMag(float agentFlowMag);

		void reSpawnReaction();
		void setReactionFeedRange(float feedMin, float feedRange);
		void setReactionColour(glm::vec3 colourA, glm::vec3 colourB);
		void setReactionFlowMag(float reactionFlowMag);

		void reloadShaders();
		void setDisplay(int display);

		int getMapWidth();
		int getMapHeight();

		void setPoints(vector<Component> points);
		int getNumPoints();

		void setCVDownScale(int downScale);
		void setBPS(float bps);

	private:
		// both
		ofShader compute_flow;
		ofShader renderer;
		ofShader simple_renderer;

		ofFbo flowFbo;

		ofShader compute_audio;
		ofFbo audioFbo;
		ofBufferObject audioBuffer;
		ofBufferObject pointsBuffer;
		int audioArraySize;

		// slime
		struct Agent{
			glm::vec2 pos;
			glm::vec2 vel;
			glm::vec4 attributes;
		};

		struct Species{
			glm::vec4 colour;
			glm::vec4 movementAttributes;
			glm::vec4 sensorAttributes;
		};

		ofShader compute_agents;
		ofShader compute_decay;
		ofShader compute_diffuse;

		vector<Agent> particles;
		ofBufferObject particlesBuffer;
		vector<Species> allSpecies;
		vector<Species> newSpecies;
		ofBufferObject allSpeciesBuffer;

		ofFbo trailFbo1;
		ofFbo trailFbo2;

		int mapWidth;
		int mapHeight;
		int mapFactor;

		float diffuseRate;
		float newDiffuseRate;
		float decayRate;
		float newDecayRate;
		float trailWeight;
		float newTrailWeight;

		// reaction diffusion
		ofShader compute_reaction;

		ofFbo reactionFbo;
		ofFbo lastReactionFbo;

		int flowSizeFactor;
		float floatStrength;

		// variables
		float time_of_day;
		float time_of_month;

		float dayRate;
		float monthRate;
		float newDayRate;
		float chemHeight;
		float newChemHeight;
		float trailHeight;
		float newTrailHeight;

		float feedMin;
		float newFeedMin;
		float feedRange;
		float newFeedRange;

		float reactionFlowMag;
		float newReactionFlowMag;
		float agentFlowMag;
		float newAgentFlowMag;

		glm::vec3 colourA;
		glm::vec3 newColourA;
		glm::vec3 colourB;
		glm::vec3 newColourB;

		vector<Component> points;
		vector<Component> newPoints;
		int numPoints = 4;

		bool bReSpawnAgents;
		int bReSpawnReaction;

		int display;

		int cvDownScale;
		float bps;
};
