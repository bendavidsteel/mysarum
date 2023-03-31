#version 440

layout(rgba8,binding=6) uniform restrict image2D trailMap;
layout(rg16,binding=7) uniform restrict image2D opticalFlowMap; // TODO switch to readonly

uniform ivec2 resolution;
uniform float deltaTime;
uniform float decayRate;
uniform int opticalFlowDownScale;

layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // accumulator
    vec4 blurredTrail = imageLoad(trailMap, coord);

    // read optical flow
    vec2 opticalFlowForce = imageLoad(opticalFlowMap, coord / opticalFlowDownScale).xy;
    opticalFlowForce = opticalFlowForce * 2. - 1.;
    float opticalFlowMag = length(opticalFlowForce);
    float opticalDecayRate = 1. - opticalFlowMag;
    
    vec4 newTrail = min(max((blurredTrail * opticalDecayRate * decayRate * deltaTime), 0.), 1.);
	imageStore(trailMap, coord, newTrail);
}