#version 440

layout(rgba8,binding=3) uniform restrict writeonly image3D trailMap;
layout(rgba8,binding=4) uniform restrict readonly image3D trailMapBack;

uniform ivec3 resolution;
uniform float deltaTime;
uniform float decayRate;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(){

    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);

    // accumulator
    vec4 blurredTrail = imageLoad(trailMapBack, coord);
    
    vec4 newTrail = min(max((blurredTrail * decayRate * deltaTime), 0.), 1.);
	imageStore(trailMap, coord, newTrail);
}