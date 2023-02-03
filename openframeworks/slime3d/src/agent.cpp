#include "agent.h"

//------------------------------------------------------------------
Agent::Agent(int speciesIdx, int numSpecies, AgentSpawn spawnStrategy)
{
	//the unique val allows us to set properties slightly differently for each particle
	speciesIndex = speciesIdx;
	speciesMask = toOneHot(speciesIdx, numSpecies);
	spawn(spawnStrategy);
}

void Agent::spawn(AgentSpawn spawnStrategy) {	
	
	if( spawnStrategy == RANDOM ){
		position.x = ofRandomWidth();
		position.y = ofRandomHeight();
	}else{
		position.x = ofGetWidth() / 2;
		position.y = ofGetHeight() / 2;
	}
	angle = ofRandom(0.0, 2 * PI);
}

//------------------------------------------------------------------
void Agent::ensureRebound(){
	glm::vec2 vel = getVelocity();
	glm::vec2 normal;
	bool rebound = false;
	if (position.x < 0)
	{
		normal = glm::vec2(1, 0);
		rebound = true;
	}
	if (position.x >= ofGetWidth())
	{
		normal = glm::vec2(-1, 0);
		rebound = true;
	}
	if (position.y < 0)
	{
		normal = glm::vec2(0, 1);
		rebound = true;
	}
	if (position.y >= ofGetHeight())
	{
		normal = glm::vec2(0, -1);
		rebound = true;
	}

	if (rebound)
	{
		glm::vec2 newVelocity = vel - (normal * 2 * dot(vel, normal));
		float newAngle = atan2(newVelocity.y, newVelocity.x);
		doRebound(newAngle);
	}
}

void Agent::doRebound(float randomAngle)
{
	position.x = min((float) ofGetWidth()-1, max(0.0f, position.x));
	position.y = min((float) ofGetHeight()-1, max(0.0f, position.y));
	angle = randomAngle;
}
  
glm::vec2 Agent::getVelocity()
{
	return glm::vec2(cos(angle), sin(angle));
}
  
void Agent::updatePosition(float deltaTime, float moveSpeed)
{
	glm::vec2 direction = getVelocity();
	position += direction * deltaTime * moveSpeed;
}
