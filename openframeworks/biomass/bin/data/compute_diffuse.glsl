#version 440

layout(rgba8,binding=6) uniform restrict image2D trailMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform float diffuseRate;
uniform ivec2 blurDir;

layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // accumulator
    vec4 originalTrail = imageLoad(trailMap, coord);
    
    //blur box size
    const int dim = 1;

    vec4 lastTrail = vec4(0.);
    ivec2 pointCoord = coord - blurDir;
    pointCoord = min(resolution-1, max(pointCoord, 0));
    lastTrail = imageLoad(trailMap, pointCoord);

    vec4 nextTrail = vec4(0.);
    pointCoord = coord + blurDir;
    pointCoord = min(resolution-1, max(pointCoord, 0));
    nextTrail = imageLoad(trailMap, pointCoord);

    vec4 sum = lastTrail + originalTrail + nextTrail;

    vec4 blurredTrail = sum / 3;
    float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
    blurredTrail = mix(originalTrail, blurredTrail, diffuseWeight);
    
	imageStore(trailMap, coord, blurredTrail);
}