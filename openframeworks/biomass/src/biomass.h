#pragma once

#include "ofMain.h"
#include "ofBufferObject.h"

#include "reactiondiffusion.h"
#include "physarum.h"

struct Component{
	glm::vec4 value;
};

class Biomass{

	public:
		void setup();
		void update(ofTexture opticalFlowTexture);
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
		ReactionDiffusion reactionDiffusion;
		Physarum physarum;

		// both
		ofShader compute_flow;
		ofShader plane_renderer;
		ofShader renderer;
		ofShader physarum_renderer;
		ofShader reaction_renderer;
		ofShader audio_renderer;
		ofShader simple_renderer;

		ofPlanePrimitive plane;
		ofPlanePrimitive reactionPlane;
		ofPlanePrimitive physarumPlane;
		ofPlanePrimitive audioPlane;

		ofSpherePrimitive randomSphere;

		ofCamera camera;
		ofLight light;
		ofMaterial material;

		ofFbo flowFbo;
		ofFbo trailFbo;
		ofFbo reactionFbo;

		ofShader compute_audio;
		ofFbo audioFbo;
		vector<Component> audioVector;
		ofBufferObject audioBuffer;
		ofBufferObject pointsBuffer;
		int audioArraySize;

		int mapWidth;
		int mapHeight;
		int mapFactor;

		int flowSizeFactor;
		float floatStrength;

		// variables
		float timeOfDay;
		float timeOfMonth;

		float dayRate;
		float monthRate;
		float newDayRate;
		float chemHeight;
		float newChemHeight;
		float trailHeight;
		float newTrailHeight;

		glm::vec3 colourA;
		glm::vec3 newColourA;
		glm::vec3 colourB;
		glm::vec3 newColourB;

		vector<Component> points;
		vector<Component> newPoints;
		int numPoints = 4;

		int display;

		int cvDownScale;
		float bps;
};
