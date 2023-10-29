#version 440

struct Particle{
	vec4 pos;
	vec4 vel;
	vec4 attr;
	vec4 color;
};

layout(std140, binding=0) buffer particle{
    Particle p[];
};

layout(std140, binding=1) buffer particleBack{
    Particle p2[];
};

layout(std140, binding=2) buffer indices{
	uint idx[];
};

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

void doRebound(inout vec3 pos)
{
	pos = min(resolution, max(pos, 0.0));
}

void ensureRebound(inout vec3 pos, inout vec3 vel){
	vec3 normal;
	bool rebound = false;
	if (pos.x < 0) {
		normal = vec3(1, 0, 0);
		rebound = true;
	} else if (pos.x >= resolution.x) {
		normal = vec3(-1, 0, 0);
		rebound = true;
	}

	if (pos.y < 0) {
		normal = vec3(0, 1, 0);
		rebound = true;
	} else if (pos.y >= resolution.y) {
		normal = vec3(0, -1, 0);
		rebound = true;
	}

	if (pos.z < 0) {
		normal = vec3(0, 0, 1);
		rebound = true;
	} else if (pos.z >= resolution.z) {
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
	vec3 pos = p2[gl_GlobalInvocationID.x].pos.xyz;
	vec3 vel = p2[gl_GlobalInvocationID.x].vel.xyz;
	float freq = p2[gl_GlobalInvocationID.x].attr.x;

	vec3 acc = vec3(0.);

	vec4 randSeed = vec4(pos.x, pos.y, pos.z, time);
	vec3 randForce = vec3(random(randSeed.xyzw), random(randSeed.yzwx), random(randSeed.zwxy));
	randForce = randForce * 2.0 - 1.0;

	acc += randForce * randomStrength;
	// acc.z = 0.0;

	vec3 color = vec3(1.0);

	float kuramotoSum = 0.0;
	int kuramotoCount = 0;

	for (int i = 0; i < numAgents; i++) {
		if (i == gl_GlobalInvocationID.x) continue;

		vec3 theirPos = p2[i].pos.xyz;
		if (angle(vel, theirPos - pos) < fov) continue;

		vec3 theirVel = p2[i].vel.xyz;
		// TODO use quaternions instead of force for direction nudges
		acc += repulse(pos, theirPos) * repulsion;
		acc += align(pos, theirPos, vel, theirVel) * alignment;
		acc += attract(pos, theirPos) * attraction;

		vec3 relPos = theirPos - pos;
		float sqd = dot(relPos, relPos);
		if (sqd < pow(kuramotoMaxDist, 2.0)) {
			float theirFreq = p2[i].attr.x;
			kuramotoSum += sin(theirFreq - freq);
			kuramotoCount++;
		}
	}

	if (kuramotoCount > 0) {
		freq += kuramotoStrength * kuramotoSum / kuramotoCount;
	}

	vec3 newVel = vel + acc * timeDelta;
	// limit to max speed
	// if (length(newVel) > maxSpeed) {
	newVel = normalize(newVel) * maxSpeed;
	// }
	vec3 newPos = pos + newVel * timeDelta;

	ensureRebound(newPos, newVel);

	p[gl_GlobalInvocationID.x].pos.xyz = newPos;
	p[gl_GlobalInvocationID.x].vel.xyz = newVel;

	p[gl_GlobalInvocationID.x].attr.x = freq;

	p[gl_GlobalInvocationID.x].color.rgb = color;
	p[gl_GlobalInvocationID.x].color.a = 0.75 + 0.25 * sin(time * freq);
}