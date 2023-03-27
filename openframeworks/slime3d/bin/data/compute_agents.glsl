#version 440

struct Agent{
	vec4 pos;
	vec4 vel;
	vec4 attributes;
};

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=0) buffer particle{
    Agent agents[];
};

layout(std140, binding=1) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=2) uniform restrict image3D trailMap;
layout(rgba8,binding=3) uniform restrict image3D flowMap;

uniform ivec3 resolution;
uniform float time;
uniform float deltaTime;
uniform float trailWeight;

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

void getOrthoVecs(vec3 vec, out vec3 vecA, out vec3 vecB) {
	vecA = vec3(0.);
	vecB = vec3(0.);
	if (vec.x != 0) {
		vecA.y = vec.x;
		vecA.x = -1 * vec.y;
		vecB.z = vec.x;
		vecB.x = -1 * vec.z;
	} else if (vec.y != 0) {
		vecA.z = vec.y;
		vecA.y = -1 * vec.z;
		vecB.x = vec.y;
		vecB.y = -1 * vec.x;
	} else {
		vecA.x = vec.z;
		vecA.z = -1 * vec.x;
		vecB.y = vec.z;
		vecB.z = -1 * vec.y;
	}
	vecA = normalize(vecA);
	vecB = normalize(vecB);
}

void getSenseVecs(vec3 pos, vec3 vel, float sensorDist, float sensorOffset, float sensorOffDist, out vec3 vecAhead, out vec3 vecA, out vec3 vecB, out vec3 vecC, out vec3 vecD) {
	vec3 velNorm = normalize(vel);
	vecAhead = velNorm * sensorDist;
	
	vec3 vecX;
	vec3 vecY;
	getOrthoVecs(vel, vecX, vecY);

	vecA = (velNorm * sensorOffDist) + (vecX * sensorOffset);
	vecB = (velNorm * sensorOffDist) + (vecX * -1 * sensorOffset);
	vecC = (velNorm * sensorOffDist) + (vecY * sensorOffset);
	vecD = (velNorm * sensorOffDist) + (vecY * -1 * sensorOffset);
}

void sense(vec3 pos, vec4 speciesMask, vec3 vecAhead, vec3 vecA, vec3 vecB, vec3 vecC, vec3 vecD, out float weightAhead, out float weightA, out float weightB, out float weightC, out float weightD) {
	
	vec3 senseAhead = pos + vecAhead;
	vec3 senseA = pos + vecA;
	vec3 senseB = pos + vecB;
	vec3 senseC = pos + vecC;
	vec3 senseD = pos + vecD;

	// TODO can we optimize in glsl?
	vec4 senseWeight = (speciesMask * 2.0) - 1.0;

	ivec3 coordAhead = min(resolution, max(ivec3(senseAhead.xyz), 0));
	vec4 trailAhead = imageLoad(trailMap, coordAhead);
	weightAhead = dot(senseWeight, trailAhead);

	ivec3 coordA = min(resolution, max(ivec3(senseA.xyz), 0));
	vec4 trailA = imageLoad(trailMap, coordA);
	weightA = dot(senseWeight, trailA);

	ivec3 coordB = min(resolution, max(ivec3(senseB.xyz), 0));
	vec4 trailB = imageLoad(trailMap, coordB);
	weightB = dot(senseWeight, trailB);

	ivec3 coordC = min(resolution, max(ivec3(senseC.xyz), 0));
	vec4 trailC = imageLoad(trailMap, coordC);
	weightC = dot(senseWeight, trailC);

	ivec3 coordD = min(resolution, max(ivec3(senseD.xyz), 0));
	vec4 trailD = imageLoad(trailMap, coordD);
	weightD = dot(senseWeight, trailD);
}

void doRebound(inout vec3 pos)
{
	pos = min(resolution-1, max(pos, 0.0));
}

void ensureRebound(inout vec3 pos, inout vec3 vel){
	vec3 normal;
	bool rebound = false;
	if (pos.x < 0) {
		normal = vec3(1, 0, 0);
		rebound = true;
	}
	if (pos.x >= resolution.x) {
		normal = vec3(-1, 0, 0);
		rebound = true;
	}
	if (pos.y < 0) {
		normal = vec3(0, 1, 0);
		rebound = true;
	}
	if (pos.y >= resolution.y) {
		normal = vec3(0, -1, 0);
		rebound = true;
	}
	if (pos.z < 0) {
		normal = vec3(0, 0, 1);
		rebound = true;
	}
	if (pos.z >= resolution.z) {
		normal = vec3(0, 0, -1);
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
	vec3 pos = agents[gl_GlobalInvocationID.x].pos.xyz;
	vec3 vel = agents[gl_GlobalInvocationID.x].vel.xyz;
	int speciesIdx = int(agents[gl_GlobalInvocationID.x].attributes.x);
	vec4 speciesMask = toMask(speciesIdx);

	// Steer based on sensory data
	float sensorDist = allSpecies[speciesIdx].sensorAttributes.x;
	float sensorOffset = allSpecies[speciesIdx].sensorAttributes.y;
	float sensorOffDist = allSpecies[speciesIdx].sensorAttributes.z;
	float moveSpeed = allSpecies[speciesIdx].movementAttributes.x;
	float turnStrength = allSpecies[speciesIdx].movementAttributes.y;

	vec3 vecAhead, vecA, vecB, vecC, vecD;
	getSenseVecs(pos, vel, sensorDist, sensorOffset, sensorOffDist, vecAhead, vecA, vecB, vecC, vecD);
	float weightAhead, weightA, weightB, weightC, weightD;
	sense(pos, speciesMask, vecAhead, vecA, vecB, vecC, vecD, weightAhead, weightA, weightB, weightC, weightD);
	
	float randomForceStrength = random(vec4(pos, time));
	vec3 agentUid = pos * time * gl_GlobalInvocationID.x;
	float randomX = (random(vec4(agentUid, vel.x)) * 2) - 1;
	float randomY = (random(vec4(agentUid, vel.y)) * 2) - 1;
	float randomZ = (random(vec4(agentUid, vel.z)) * 2) - 1;
	vec3 randomForce = normalize(vec3(randomX, randomY, randomZ));

	vec3 force = randomForce * 0.01;
	if (weightAhead < weightA && weightAhead < weightB && weightAhead < weightC && weightAhead < weightD) {
		force += randomForce;
	} else if (weightA > weightAhead && weightA > weightB && weightA > weightC && weightA > weightD) {
		force += normalize(vecA);
	} else if (weightB > weightAhead && weightB > weightA && weightB > weightC && weightB > weightD) {
		force += normalize(vecB);
	} else if (weightC > weightAhead && weightC > weightA && weightC > weightB && weightC > weightD) {
		force += normalize(vecC);
	} else if (weightD > weightAhead && weightD > weightA && weightD > weightB && weightD > weightC) {
		force += normalize(vecD);
	} else if (weightAhead > weightA && weightAhead > weightB && weightAhead > weightC && weightAhead > weightD) {
		force += vec3(0.);
	}

	// pushing agents around based on flow
	ivec3 oldCoord = ivec3(pos.xyz);
	vec3 flowForce = imageLoad(flowMap, oldCoord).xyz;
	flowForce = (2 * flowForce) - 1; // convert to -1-1 range
	force += 2 * flowForce;

	// Update position
	vec3 newVel = normalize(vel + (force * randomForceStrength * turnStrength * deltaTime));
	vec3 newPos = pos + (newVel * deltaTime * moveSpeed);

	// Clamp position to map boundaries, and pick new random move dir if hit boundary
	ensureRebound(newPos, newVel);

	agents[gl_GlobalInvocationID.x].vel.xyz = newVel;
	agents[gl_GlobalInvocationID.x].pos.xyz = newPos;

	ivec3 newCoord = ivec3(newPos.xyz);
	vec4 oldTrail = imageLoad(trailMap, newCoord);
	vec4 newTrail = max(min((oldTrail + (speciesMask * trailWeight * deltaTime)), 1.), 0.);
	imageStore(trailMap, newCoord, newTrail);
}