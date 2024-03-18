#version 440

struct Particle{
	vec4 pos;
	vec4 vel;
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

uniform ivec2 resolution;
uniform int numAgents;
uniform int numCompare;
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
		return dir;
	}
	return vec3(0.0);
} 

vec3 align(vec3 my_pos, vec3 their_pos, vec3 my_vel, vec3 their_vel){
	vec3 d = their_pos - my_pos;
	vec3 dv = their_vel - my_vel;
	if (dot(d,d) < pow(alignmentMaxDist, 2.0)){
		return dv / (dot(dv,dv) + 10.0);
	}
	return vec3(0.0);
}

vec3 attract(vec3 my_pos, vec3 their_pos){
	vec3 dir = their_pos-my_pos;
	float sqd = dot(dir,dir);
	if(sqd < pow(attractionMaxDist,2.0)){
		float f = 1000000.0/sqd;
		return normalize(dir)*f;
	}
	return vec3(0.0);
}

vec2 vecFromAngle(float angle){
	return vec2(cos(angle), sin(angle));
}

void doRebound(inout vec2 pos)
{
	pos = min(resolution, max(pos, 0.0));
}

void ensureRebound(inout vec2 pos, inout vec2 vel){
	vec2 normal;
	bool rebound = false;
	if (pos.x < 0)
	{
		normal = vec2(1, 0);
		rebound = true;
	}
	if (pos.x >= resolution.x)
	{
		normal = vec2(-1, 0);
		rebound = true;
	}
	if (pos.y < 0)
	{
		normal = vec2(0, 1);
		rebound = true;
	}
	if (pos.y >= resolution.y)
	{
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
	vec2 pos = p2[gl_GlobalInvocationID.x].pos.xy;
	vec2 vel = p2[gl_GlobalInvocationID.x].vel.xy;

	float theta = atan(vel.y, vel.x);

	vec3 randSeed = vec3(pos.x, pos.y, time);
	float randomTheta = (2. * random(randSeed)) - 1.;

	float newTheta = theta + randomTheta * randomStrength;

	for (int i = 0; i < numAgents; i++) {
		
	}

	vec2 newVel = vec2(cos(newTheta), sin(newTheta)) * maxSpeed;
	vec2 newPos = pos + newVel * timeDelta;

	ensureRebound(newPos, newVel);

	p[gl_GlobalInvocationID.x].pos.xy = newPos;
	p[gl_GlobalInvocationID.x].pos.z = 0.;

	p[gl_GlobalInvocationID.x].vel.xy = newVel;

	p[gl_GlobalInvocationID.x].color.rgb = vec3(1.);
	p[gl_GlobalInvocationID.x].color.a = 1.;
}