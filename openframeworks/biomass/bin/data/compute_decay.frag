#version 440

layout(rg16,binding=4) uniform restrict image2D optFlowMap; // TODO switch to readonly

uniform sampler2DRect trailMap;
uniform sampler2DRect particleMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform float decayRate;
uniform int opticalFlowDownScale;

out vec4 out_color;

void main(){

    ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 oldTrail = imageLoad(trailMap, newCoord);
	vec4 newTrail = max(min((oldTrail + (speciesMask * trailWeight * deltaTime)), 1.), 0.);
	imageStore(trailMap, newCoord, newTrail);

    // accumulator
    vec4 blurredTrail = texture(trailMap, coord);

    // read optical flow
    vec2 opticalFlowForce = imageLoad(optFlowMap, coord / opticalFlowDownScale).xy;
    opticalFlowForce = opticalFlowForce * 2. - 1.;
    float opticalFlowMag = length(opticalFlowForce);
    float opticalDecayRate = 1. - 0.8 * opticalFlowMag;
    
    vec4 newTrail = min(max((blurredTrail * opticalDecayRate * decayRate * deltaTime), 0.), 1.);

    if (newTrail.x < 0.001) {
        newTrail.x = 0.;
    }
    if (newTrail.y < 0.001) {
        newTrail.y = 0.;
    }
    if (newTrail.z < 0.001) {
        newTrail.z = 0.;
    }
    if (newTrail.w < 0.001) {
        newTrail.w = 0.;
    }

    out_color = newTrail;
}