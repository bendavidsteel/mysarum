#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;
layout(rg16,binding=1) uniform restrict image2D trailMap;

uniform ivec2 resolution;
uniform float deltaTime;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(1.0) * vec2(coord) / vec2(resolution);

    float reaction = 0;
    if (distance(uv, vec2(0.25, 0.25)) < 0.1) {
        reaction = 1;
    }

    if (distance(uv, vec2(0.75, 0.75)) < 0.1) {
        reaction = 1;
    }
    
    vec4 vals = vec4(1 - reaction, reaction, 0., 1.);
	imageStore(reactionMap, coord, vals);

    float trail = 0;
    if (distance(uv, vec2(0.75, 0.25)) < 0.1) {
        trail = 1;
    }
    vals = vec4(trail, 0., 0., 1.);
    imageStore(trailMap, coord, vals);
}