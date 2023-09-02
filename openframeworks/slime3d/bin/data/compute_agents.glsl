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


vec3 vecFromThetaPhi(float theta, float phi) {
	return vec3(cos(phi) * cos(theta), cos(phi) * sin(theta), sin(phi));
}

float sense(vec3 pos, float theta, float phi, vec4 speciesMask, float sensorOffsetDist, float sensorThetaOffset, float sensorPhiOffset) {
	float sensorTheta = theta + sensorThetaOffset;
	float sensorPhi = phi + sensorPhiOffset;
	vec3 sensorDir = vecFromThetaPhi(sensorTheta, sensorPhi);

	vec3 sensorPos = pos + (sensorDir * sensorOffsetDist);
	int sensorCentreX = int(sensorPos.x);
	int sensorCentreY = int(sensorPos.y);
	int sensorCentreZ = int(sensorPos.z);

	float sum = 0;
	const int sensorSize = 0;

	// TODO can we optimize in glsl?
	vec4 senseWeight = (speciesMask * 2.0) - 1.0;

	ivec3 sampleCoord = min(resolution, max(ivec3(sensorCentreX, sensorCentreY, sensorCentreZ), 0));
	vec4 trailWeight = imageLoad(trailMap, sampleCoord);
	sum += dot(senseWeight, trailWeight);

	return sum;
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
	float sensorAngleRad = allSpecies[speciesIdx].sensorAttributes.x;
	float sensorOffsetDist = allSpecies[speciesIdx].sensorAttributes.y;
	float moveSpeed = allSpecies[speciesIdx].movementAttributes.x;
	float turnSpeed = allSpecies[speciesIdx].movementAttributes.y;

	float theta = atan(vel.y, vel.x);
	float phi = atan(vel.z, sqrt(vel.x * vel.x + vel.y * vel.y));

	float weightAhead = sense(pos, theta, phi, speciesMask, sensorOffsetDist, 0, 0);
	float weightLeft = sense(pos, theta, phi, speciesMask, sensorOffsetDist, sensorAngleRad, 0);
	float weightRight = sense(pos, theta, phi, speciesMask, sensorOffsetDist, -sensorAngleRad, 0);
	float weightUp = sense(pos, theta, phi, speciesMask, sensorOffsetDist, 0, sensorAngleRad);
	float weightDown = sense(pos, theta, phi, speciesMask, sensorOffsetDist, 0, -sensorAngleRad);
	
	float randomThetaStrength = random(vec4(pos, time));
	float randomPhiStrength = random(vec4(pos, time));

	if (weightAhead < weightLeft && weightAhead < weightRight && weightAhead < weightUp && weightAhead < weightDown) {
		theta += (randomThetaStrength - 0.5) * 2 * turnSpeed * deltaTime;
		phi += (randomPhiStrength - 0.5) * 2 * turnSpeed * deltaTime;
	} else if (weightLeft > weightAhead && weightLeft > weightRight && weightLeft > weightUp && weightLeft > weightDown) {
		theta += randomThetaStrength * turnSpeed * deltaTime;
		phi += (randomPhiStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	} else if (weightRight > weightAhead && weightRight > weightLeft && weightRight > weightUp && weightRight > weightDown) {
		theta -= randomThetaStrength * turnSpeed * deltaTime;
		phi += (randomPhiStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	} else if (weightUp > weightAhead && weightUp > weightLeft && weightUp > weightRight && weightUp > weightDown) {
		phi += randomPhiStrength * turnSpeed * deltaTime;
		theta += (randomThetaStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	} else if (weightDown  > weightAhead && weightDown > weightLeft && weightDown > weightRight && weightDown > weightUp) {
		phi -= randomPhiStrength * turnSpeed * deltaTime;
		theta += (randomThetaStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	} else if (weightAhead > weightLeft && weightAhead > weightRight && weightAhead > weightUp && weightAhead > weightDown) {
		theta += (randomThetaStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
		phi += (randomPhiStrength - 0.5) * 0.5 * turnSpeed * deltaTime;
	}

	vel = vecFromThetaPhi(theta, phi);

	// pushing agents around based on flow
	ivec3 oldCoord = ivec3(pos.xyz);
	vec3 flowForce = imageLoad(flowMap, oldCoord).xyz;
	flowForce = (2 * flowForce) - 1; // convert to -1-1 range
	vec3 force = 0.1 * flowForce;

	// Update position
	vec3 newVel = normalize(vel + (force * deltaTime));
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