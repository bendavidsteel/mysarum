#version 440

struct Agent{
	vec2 pos;
	vec2 vel;
	vec4 speciesMask;
};

struct Species{
	vec4 colour;
	vec4 movementAttributes;
	vec4 sensorAttributes;
};

layout(std140, binding=1) buffer agents{
    Agent allAgents[];
};

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

layout(rg16,binding=3) uniform restrict image2D agentMap;

uniform sampler2DRect flowMap;
uniform sampler2DRect reactionMap;
uniform sampler2DRect audioMap;
uniform sampler2DRect optFlowMap;
uniform sampler2DRect trailMap;

uniform ivec2 resolution;
uniform float time;
uniform float deltaTime;
uniform float trailWeight;
uniform int opticalFlowDownScale;
uniform float agentFlowMag;

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
	x += ( x << 10u );
	x ^= ( x >>  6u );
	x += ( x <<  3u );
	x ^= ( x >> 11u );
	x += ( x << 15u );
	return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct agent float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

vec4 toMask(int idx) {
	vec4 ret = vec4(0.0);
	if (idx == 0) {
		ret.x = 1.0;
	} else if (idx == 1) {
		ret.y = 1.0;
	} else if (idx == 2) {
		ret.z = 1.0;
	} else if (idx == 3) {
		ret.w = 1.0;
	}
	return ret;
}


vec2 vecFromAngle(float angle) {
	return vec2(cos(angle), sin(angle));
}

float sense(vec2 pos, float angle, vec4 speciesMask, float sensorOffsetDist, float sensorAngleOffset) {
	float sensorAngle = angle + sensorAngleOffset;
	vec2 sensorDir = vecFromAngle(sensorAngle);

	vec2 sensorPos = pos + (sensorDir * sensorOffsetDist);
	int sensorCentreX = int(sensorPos.x);
	int sensorCentreY = int(sensorPos.y);

	float sum = 0;
	const int sensorSize = 0;

	// TODO can we optimize in glsl?
	vec4 senseWeight = (speciesMask * 2.0) - 1.0;
	for (int offsetX = -sensorSize; offsetX <= sensorSize; offsetX ++) {
		for (int offsetY = -sensorSize; offsetY <= sensorSize; offsetY ++) {
			ivec2 sampleCoord = min(resolution, max(ivec2(sensorCentreX + offsetX, sensorCentreY + offsetY), 0));
			vec4 trailWeight = texture(trailMap, sampleCoord);
			sum += dot(senseWeight, trailWeight);

			// repel from optical flow
			vec2 opticalFlow = texture(optFlowMap, sampleCoord / opticalFlowDownScale).xy;
			opticalFlow = opticalFlow * 2.0 - 1.0;
			float opticalFlowMag = length(opticalFlow);
			sum -= 10 * opticalFlowMag;

			// repel from reaction peaks
			float chem_y = texture(reactionMap, sampleCoord).y;
			sum -= 20 * chem_y;
		}
	}

	return sum;
}

void doRebound(inout vec2 pos)
{
	pos = min(resolution-1, max(pos, 0.0));
}

void ensureRebound(inout vec2 pos, inout vec2 vel){
	vec2 normal;
	bool rebound = false;
	if (pos.x < 0) {
		normal = vec2(1, 0);
		rebound = true;
	}
	if (pos.x >= resolution.x) {
		normal = vec2(-1, 0);
		rebound = true;
	}
	if (pos.y < 0) {
		normal = vec2(0, 1);
		rebound = true;
	}
	if (pos.y >= resolution.y) {
		normal = vec2(0, -1);
		rebound = true;
	}

	if (rebound)
	{
		vel = vel - (normal * 2 * dot(vel, normal));
		doRebound(pos);
	}
}


layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
void main(){
	vec2 pos = allAgents[gl_GlobalInvocationID.x].pos.xy;
	vec2 vel = allAgents[gl_GlobalInvocationID.x].vel.xy;
	vec4 speciesMask = allAgents[gl_GlobalInvocationID.x].speciesMask;

	int speciesIdx = 0;
	if (speciesMask.x == 1.0) {
		speciesIdx = 0;
	} else if (speciesMask.y == 1.0) {
		speciesIdx = 1;
	} else if (speciesMask.z == 1.0) {
		speciesIdx = 2;
	} else if (speciesMask.w == 1.0) {
		speciesIdx = 3;
	}

	// Steer based on sensory data
	float sensorAngleRad = allSpecies[speciesIdx].sensorAttributes.x;
	float sensorOffsetDist = allSpecies[speciesIdx].sensorAttributes.y;
	float moveSpeed = allSpecies[speciesIdx].movementAttributes.x;
	float turnSpeed = allSpecies[speciesIdx].movementAttributes.y;

	float angle = atan(vel.y, vel.x);

	float weightForward = sense(pos, angle, speciesMask, sensorOffsetDist, 0);
	float weightLeft = sense(pos, angle, speciesMask, sensorOffsetDist, sensorAngleRad);
	float weightRight = sense(pos, angle, speciesMask, sensorOffsetDist, -sensorAngleRad);

	float randomSteerStrength = random(vec3(pos, time));

	if (weightForward < weightLeft && weightForward < weightRight) {
		angle += (randomSteerStrength - 0.5) * 2 * turnSpeed * deltaTime;
	}
	// Turn right
	else if (weightRight > weightLeft) {
		angle -= randomSteerStrength * turnSpeed * deltaTime;
	}
	// Turn left
	else if (weightLeft > weightRight) {
		angle += randomSteerStrength * turnSpeed * deltaTime;
	} else {
		angle += (randomSteerStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	}

	vel = vecFromAngle(angle);

	// pushing agents around based on flow
	ivec2 oldCoord = ivec2(pos.xy);
	vec2 simplexFlowForce = texture(flowMap, oldCoord).xy;
	simplexFlowForce = (2 * simplexFlowForce) - 1; // convert to -1-1 range

	vec2 opticalFlowForce = texture(optFlowMap, oldCoord / opticalFlowDownScale).xy;
	opticalFlowForce = (2 * opticalFlowForce) - 1; // convert to -1-1 range
	float opticalFlowMag = length(opticalFlowForce);

	float audioMag = texture(audioMap, oldCoord).x;

	// vec2 force = ((2 * opticalFlowMag) + (0.5 * audioMag) + agentFlowMag) * simplexFlowForce + opticalFlowForce;
	vec2 force = vec2(0.);

	// Update position
	vec2 newVel = normalize(vel + (force * deltaTime));
	vec2 newPos = pos + (newVel * deltaTime * moveSpeed);;

	// Clamp position to map boundaries, and pick new random move dir if hit boundary
	ensureRebound(newPos, newVel);

	allAgents[gl_GlobalInvocationID.x].vel.xy = newVel;
	allAgents[gl_GlobalInvocationID.x].pos.xy = newPos;

	ivec2 newCoord = ivec2(newPos.xy);
	imageStore(agentMap, newCoord, speciesMask);
}