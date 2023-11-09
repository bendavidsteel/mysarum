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

layout(std140, binding=0) restrict writeonly buffer particle{
    Agent agents[];
};

layout(std140, binding=1) restrict readonly buffer particleBack{
	Agent agents2[];
};

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=3) uniform restrict writeonly image3D trailMap;
layout(rgba8,binding=4) uniform restrict readonly image3D trailMapBack;

uniform ivec3 resolution;
uniform float time;
uniform float deltaTime;
uniform float trailWeight;

uniform float sensorAngleRad;
uniform float sensorOffsetDist;
uniform float moveSpeed;
uniform float turnSpeed;

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


vec3 turnTowardVector(vec3 fromVector, vec3 toVector, float frac) {
    // Calculate the dot product
    float dotProduct = dot(fromVector, toVector);
    
    // Calculate the axis of rotation (cross product)
    vec3 axis = normalize(cross(fromVector, toVector));
	vec3 perpComponent = normalize(cross(axis, fromVector));
    
    // Calculate the angle of rotation
    float angle = acos(dotProduct / (length(fromVector) * length(toVector)));

	// apply fractional rotation
	angle *= frac;
    
    // get vector rotated by angle around axis
	vec3 rotatedVector = fromVector * cos(angle) + perpComponent * sin(angle);
    
    return rotatedVector;
}

vec3 senseDir(vec3 velNorm, vec3 orthoA, vec3 orthoB, float sensorOffsetDist, float orthoAOffset, float orthoBOffset) {
	vec3 sensorDir = velNorm * sensorOffsetDist;

	if (orthoAOffset > 0) {
		sensorDir += orthoA * sensorOffsetDist * tan(orthoAOffset);
	} else if (orthoAOffset < 0) {
		sensorDir -= orthoA * sensorOffsetDist * tan(-orthoAOffset);
	}

	if (orthoBOffset > 0) {
		sensorDir += orthoB * sensorOffsetDist * tan(orthoBOffset);
	} else if (orthoBOffset < 0) {
		sensorDir -= orthoB * sensorOffsetDist * tan(-orthoBOffset);
	}

	return sensorDir;
}

float sense(vec3 pos, vec3 sensorDir, vec4 speciesMask) {
	vec3 sensorPos = pos + sensorDir;

	int sensorCentreX = int(sensorPos.x);
	int sensorCentreY = int(sensorPos.y);
	int sensorCentreZ = int(sensorPos.z);

	float sum = 0;

	// TODO can we optimize in glsl?
	vec4 senseWeight = (speciesMask * 2.0) - 1.0;

	ivec3 sampleCoord = min(resolution, max(ivec3(sensorCentreX, sensorCentreY, sensorCentreZ), 0));
	vec4 trailWeight = imageLoad(trailMapBack, sampleCoord);
	sum += dot(senseWeight, trailWeight);

	return sum;
}

void doRebound(inout vec3 pos)
{
	pos = min(resolution, max(pos, 0.0));
}

void ensureRebound(inout vec3 pos, inout vec3 vel){
	vec3 normal;
	bool rebound = false;
	if (pos.x < 0.) {
		normal = vec3(1, 0, 0);
		rebound = true;
	}
	if (pos.x >= resolution.x) {
		normal = vec3(-1, 0, 0);
		rebound = true;
	}
	if (pos.y < 0.) {
		normal = vec3(0, 1, 0);
		rebound = true;
	}
	if (pos.y >= resolution.y) {
		normal = vec3(0, -1, 0);
		rebound = true;
	}
	if (pos.z < 0.) {
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
	vec3 pos = agents2[gl_GlobalInvocationID.x].pos.xyz;
	vec3 vel = agents2[gl_GlobalInvocationID.x].vel.xyz;
	int speciesIdx = int(agents2[gl_GlobalInvocationID.x].attributes.x);
	vec4 speciesMask = toMask(speciesIdx);

	// Steer based on sensory data
	// float sensorAngleRad = allSpecies[speciesIdx].sensorAttributes.x;
	// float sensorOffsetDist = allSpecies[speciesIdx].sensorAttributes.y;
	// float moveSpeed = allSpecies[speciesIdx].movementAttributes.x;
	// float turnSpeed = allSpecies[speciesIdx].movementAttributes.y;

	vec4 randSeed = vec4(pos.x, pos.y, pos.z, time);
	vec3 randVec = (2. * vec3(random(randSeed.xyzw), random(randSeed.yzwx), random(randSeed.zwxy))) - 1.;

	vec3 orthoA = normalize(cross(vel, randVec));
	vec3 orthoB = normalize(cross(vel, orthoA));
	vec3 velNorm = normalize(vel);

	vec3 aheadDir = senseDir(velNorm, orthoA, orthoB, sensorOffsetDist, 0, 0);
	vec3 leftDir = senseDir(velNorm, orthoA, orthoB, sensorOffsetDist, sensorAngleRad, 0);
	vec3 rightDir = senseDir(velNorm, orthoA, orthoB, sensorOffsetDist, -sensorAngleRad, 0);
	vec3 upDir = senseDir(velNorm, orthoA, orthoB, sensorOffsetDist, 0, sensorAngleRad);
	vec3 downDir = senseDir(velNorm, orthoA, orthoB, sensorOffsetDist, 0, -sensorAngleRad);

	float weightAhead = sense(pos, aheadDir, speciesMask);
	float weightLeft = sense(pos, leftDir, speciesMask);
	float weightRight = sense(pos, rightDir, speciesMask);
	float weightUp = sense(pos, upDir, speciesMask);
	float weightDown = sense(pos, downDir, speciesMask);

	// get random perpendicular direction to vel
	vec3 randPerp = normalize(cross(velNorm, randVec));

	float randomAngle = random(randSeed) * turnSpeed;

	vec3 newVel = vel;

	if (weightLeft > weightAhead && weightLeft > weightRight && weightLeft > weightUp && weightLeft > weightDown) {
		newVel = turnTowardVector(velNorm, normalize(leftDir), randomAngle);
	} else if (weightRight > weightAhead && weightRight > weightLeft && weightRight > weightUp && weightRight > weightDown) {
		newVel = turnTowardVector(velNorm, normalize(rightDir), randomAngle);
	} else if (weightUp > weightAhead && weightUp > weightLeft && weightUp > weightRight && weightUp > weightDown) {
		newVel = turnTowardVector(velNorm, normalize(upDir), randomAngle);
	} else if (weightDown > weightAhead && weightDown > weightLeft && weightDown > weightRight && weightDown > weightUp) {
		newVel = turnTowardVector(velNorm, normalize(downDir), randomAngle);
	} else if (weightAhead > weightLeft && weightAhead > weightRight && weightAhead > weightUp && weightAhead > weightDown) {
		randomAngle /= 4.;
		newVel = turnTowardVector(velNorm, randPerp, randomAngle);
	} else if (weightAhead < weightLeft && weightAhead < weightRight && weightAhead < weightUp && weightAhead < weightDown) {
		newVel = turnTowardVector(velNorm, randPerp, randomAngle);
	} else {
		randomAngle /= 2.;
		newVel = turnTowardVector(velNorm, randPerp, randomAngle);
	}

	float maxSense = max(max(max(max(weightAhead, weightLeft), weightRight), weightUp), weightDown);
	float avgSense = (weightAhead + weightLeft + weightRight + weightUp + weightDown) / 5.;
	float turnAmount = randomAngle / turnSpeed;

	// pushing agents around based on flow
	// ivec3 oldCoord = ivec3(pos.xyz);
	// vec3 flowForce = imageLoad(flowMap, oldCoord).xyz;
	// flowForce = (2 * flowForce) - 1; // convert to -1-1 range
	vec3 gravForce = vec3(0, 0.01, 0.);
	vec3 force = gravForce;

	// Update position
	newVel = normalize(newVel + (force * deltaTime));
	vec3 newPos = pos + (newVel * deltaTime * moveSpeed);

	// Clamp position to map boundaries, and pick new random move dir if hit boundary
	ensureRebound(newPos, newVel);

	agents[gl_GlobalInvocationID.x].vel.xyz = newVel;
	agents[gl_GlobalInvocationID.x].pos.xyz = newPos;
	agents[gl_GlobalInvocationID.x].state.xyz = vec3(maxSense, avgSense, turnAmount);

	ivec3 newCoord = ivec3(newPos.xyz);
	vec4 oldTrail = imageLoad(trailMapBack, newCoord);
	vec4 newTrail = max(min((oldTrail + (speciesMask * trailWeight * deltaTime)), 1.), 0.);
	imageStore(trailMap, newCoord, newTrail);
}