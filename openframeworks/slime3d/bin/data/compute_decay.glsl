#version 440

layout(rgba8,binding=2) uniform restrict image3D trailMap;

uniform ivec3 resolution;
uniform float deltaTime;
uniform float decayRate;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
void main(){

    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);

    // accumulator
    vec4 blurredTrail = imageLoad(trailMap, coord);
    
    vec4 newTrail = min(max((blurredTrail * decayRate * deltaTime), 0.), 1.);
	imageStore(trailMap, coord, newTrail);
}