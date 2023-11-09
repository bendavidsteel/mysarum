#version 440

layout(rgba8,binding=3) uniform restrict writeonly image3D trailMap;
layout(rgba8,binding=4) uniform restrict readonly image3D trailMapBack;

uniform ivec3 resolution;
uniform float deltaTime;
uniform float diffuseRate;
uniform ivec3 blurDir;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(){

    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);

    // accumulator
    vec4 originalTrail = imageLoad(trailMapBack, coord);
    vec4 sum = vec4(0.0);

    ivec3 pointCoord = coord - blurDir;
    pointCoord = min(resolution, max(pointCoord, 0));
    sum += imageLoad(trailMapBack, pointCoord);

    sum += originalTrail;

    pointCoord = coord + blurDir;
    pointCoord = min(resolution, max(pointCoord, 0));
    sum += imageLoad(trailMapBack, pointCoord);

    vec4 blurredTrail = sum / 3;
    float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
    blurredTrail = (originalTrail * (1 - diffuseWeight)) + (blurredTrail * diffuseWeight);
    
	imageStore(trailMap, coord, blurredTrail);
}