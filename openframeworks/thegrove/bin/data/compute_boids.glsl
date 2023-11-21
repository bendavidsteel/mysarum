#version 440

struct Particle{
	vec4 pos;
	vec4 vel;
	vec4 attr;
	vec4 color;
};

layout(std140, binding=5) buffer particle{
    Particle p[];
};

layout(std140, binding=6) buffer particleBack{
    Particle p2[];
};

layout(rgba8,binding=4) uniform restrict readonly image3D trailMapBack;

// layout(std140, binding=7) buffer tree{
// 	float t[];
// };

uniform ivec3 resolution;
uniform int numAgents;
uniform float timeDelta;
uniform float time;
uniform float attraction;
uniform float attractionMaxDist;
uniform float alignment;
uniform float alignmentMaxDist;
uniform float repulsion;
uniform float repulsionMaxDist;
uniform float maxSpeed;
uniform float randomStrength;
uniform float fov;
uniform float kuramotoStrength;
uniform float kuramotoMaxDist;
uniform float windStrength;
uniform vec2 windDirection;
uniform float brightness;

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


void doRebound(inout vec3 pos)
{
	pos = min(resolution-1, max(pos, 0.0));
}

void avoidWalls(vec3 pos, inout vec3 vel){
	// float wallDist = 0.00001;
	// vel.x += resolution.x * wallDist / pos.x;
	// vel.x -= resolution.x * wallDist / (resolution.x - pos.x);

	// vel.y += resolution.y * wallDist / pos.y;
	// vel.y -= resolution.y * wallDist / (resolution.y - pos.y);

	// vel.z += resolution.z * wallDist / pos.z;
	// vel.z -= resolution.z * wallDist / (resolution.z - pos.z);

	if (pos.x < 0) {
		vel.x += 1.;
	} else if (pos.x > resolution.x) {
		vel.x -= 1.;
	}

	if (pos.y < 0) {
		vel.y += 1.;
	} else if (pos.y > resolution.y) {
		vel.y -= 1.;
	}

	if (pos.z < 0) {
		vel.z += 1.;
	} else if (pos.z > resolution.z) {
		vel.z -= 1.;
	}
}

void ensureRebound(inout vec3 pos, inout vec3 vel){
	vec3 normal;
	bool rebound = false;
	if (pos.x < 0) {
		normal = vec3(1, 0, 0);
		rebound = true;
	} else if (pos.x > resolution.x) {
		normal = vec3(-1, 0, 0);
		rebound = true;
	}

	if (pos.y < resolution.y) {
		normal = vec3(0, 1, 0);
		rebound = true;
	} else if (pos.y > resolution.y) {
		normal = vec3(0, -1, 0);
		rebound = true;
	}

	if (pos.z < 0) {
		normal = vec3(0, 0, 1);
		rebound = true;
	} else if (pos.z > resolution.z) {
		normal = vec3(0, 0, -1);
		rebound = true;
	}

	if (rebound)
	{
		vel = vel - (normal * 2 * dot(vel, normal));
		doRebound(pos);
	}
}


vec3 repulse(vec3 my_pos, vec3 their_pos){
	vec3 dir = my_pos-their_pos;
	float sqd = dot(dir,dir);
	if(sqd < pow(repulsionMaxDist,2.0)){
		return normalize(dir) * 1./sqd;
	}
	return vec3(0.0);
} 

vec3 align(vec3 my_pos, vec3 their_pos, vec3 my_vel, vec3 their_vel){
	vec3 d = their_pos - my_pos;
	vec3 dv = their_vel - my_vel;
	float sqd = dot(d,d);
	if (sqd < pow(alignmentMaxDist, 2.0)){
		return normalize(dv) * 1./sqd;
	}
	return vec3(0.0);
}

vec3 attract(vec3 my_pos, vec3 their_pos){
	vec3 dir = their_pos-my_pos;
	float sqd = dot(dir,dir);
	if(sqd < pow(attractionMaxDist,2.0)){
		float f = 1.0/sqd;
		return normalize(dir) * f;
	}
	return vec3(0.0);
}

float angle(vec3 my_vel, vec3 their_vel){
	return dot(my_vel, their_vel) / (length(my_vel) * length(their_vel));
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

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
void main(){
	vec3 pos = p2[gl_GlobalInvocationID.x].pos.xyz;
	vec3 vel = p2[gl_GlobalInvocationID.x].vel.xyz;
	float naturalFreq = p2[gl_GlobalInvocationID.x].attr.x;
	float phase = p2[gl_GlobalInvocationID.x].attr.y;

	vec3 velNorm = normalize(vel);

	vec4 randSeed = vec4(pos.x, pos.y, pos.z, time);
	vec3 randVec = vec3(random(randSeed.xyzw), random(randSeed.yzwx), random(randSeed.zwxy));
	randVec = randVec * 2.0 - 1.0;

	// get random perpendicular direction to vel
	vec3 randPerp = normalize(cross(velNorm, randVec));

	float randomAngle = random(randSeed.xyzw) * randomStrength;

	vec3 newVel = turnTowardVector(vel, randPerp, randomAngle);

	vec3 color = vec3(1.0);

	float freqSum = 0.0;
	float phaseSum = 0.0;
	int kuramotoCount = 0;

	vec3 repulseSum = vec3(0.0);
	int repulseCount = 0;

	vec3 alignSum = vec3(0.0);
	int alignCount = 0;

	vec3 attractSum = vec3(0.0);
	int attractCount = 0;

	for (int i = 0; i < numAgents; i++) {
		if (i == gl_GlobalInvocationID.x) continue;

		vec3 theirPos = p2[i].pos.xyz;
		if (angle(vel, theirPos - pos) < fov) continue;

		vec3 theirVel = p2[i].vel.xyz;

		vec3 relPos = theirPos - pos;
		float sqd = dot(relPos, relPos);

		if (sqd < pow(repulsionMaxDist, 2.0)) {
			repulseSum += relPos;
			repulseCount++;
		}

		if (sqd < pow(alignmentMaxDist, 2.0)) {
			alignSum += theirVel;
			alignCount++;
		}

		if (sqd < pow(attractionMaxDist, 2.0)) {
			attractSum += relPos;
			attractCount++;
		}

		if (sqd < pow(kuramotoMaxDist, 2.0)) {
			float theirFreq = p2[i].attr.x;
			float theirPhase = p2[i].attr.y;
			phaseSum += sin(theirPhase - phase);
			freqSum += theirFreq - naturalFreq;
			kuramotoCount++;
		}
	}
		
	if (repulseCount > 0) {
		vec3 repulseDir = repulseSum / repulseCount;
		newVel = turnTowardVector(newVel, -repulseDir, repulsion);
	}	

	if (alignCount > 0) {
		vec3 alignDir = alignSum / alignCount;
		newVel = turnTowardVector(newVel, alignDir, alignment);
	}

	if (attractCount > 0) {
		vec3 attractDir = attractSum / attractCount;
		newVel = turnTowardVector(newVel, attractDir, attraction);
	}

	// float freq = naturalFreq;
	// float phase = naturalPhase;
	if (kuramotoCount > 0) {
		// implement kuramoto
		// freq = naturalFreq + kuramotoStrength * freqSum / kuramotoCount;
		// Update phase based on the Kuramoto model
		phase += (naturalFreq + (kuramotoStrength * phaseSum / kuramotoCount)) * timeDelta;
		phase = mod(phase, 2.0 * 3.141592);
	}

	newVel = normalize(newVel) * maxSpeed;

	// avoid the physarum
	// vec3 sensorCentre = pos + velNorm * repulsionMaxDist;
	// ivec3 sampleCoord = min(resolution, max(ivec3(sensorCentre), 0));
	// vec4 trailWeight = imageLoad(trailMapBack, sampleCoord);
	// if (trailWeight.r > 0.2) {
	// 	vec3 sensorDir = sensorCentre - pos;
		// newVel += normalize(-sensorDir) * 10;// / dot(sensorDir, sensorDir);
	// }

	// avoid the tree
	// float spaceVectorFactor = 0.1;
	// int treeVectorIdx = int(pos.x * spaceVectorFactor) + int(pos.y * spaceVectorFactor * resolution.x * spaceVectorFactor) + int(pos.z * spaceVectorFactor * resolution.x * spaceVectorFactor * resolution.y * spaceVectorFactor);
	// float treePresence = t[treeVectorIdx];
	// if (treePresence > 0.) {
	// 	vec3 sensorDir = sensorCentre - pos;
		// newVel += normalize(-sensorDir) * 10;// / dot(sensorDir, sensorDir);
	// }

	avoidWalls(pos, newVel);

	// apply wind
	newVel.xz += windDirection * windStrength;

	vec3 newPos = pos + newVel * timeDelta;

	// ensureRebound(newPos, newVel);

	p[gl_GlobalInvocationID.x].pos.xyz = newPos;
	p[gl_GlobalInvocationID.x].vel.xyz = newVel;

	p[gl_GlobalInvocationID.x].attr.y = phase;

	float amp = sin((time * naturalFreq) + phase);
	p[gl_GlobalInvocationID.x].color.rgb = color;
	p[gl_GlobalInvocationID.x].color.a = (0.66 + 0.33 * amp) * brightness;
}