#version 440

layout(rg16,binding=4) uniform restrict image2D optFlowMap; // TODO switch to readonly

uniform sampler2DRect trailMap;
uniform sampler2DRect agentMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform float decayRate;
uniform float trailWeight;
uniform int opticalFlowDownScale;

out vec4 out_color;

void main(){

    vec2 coord = gl_FragCoord.xy;

    vec4 oldTrail = texture(trailMap, coord);
    vec4 speciesMask = texture(agentMap, coord);
	vec4 newTrail = max(min((oldTrail + (speciesMask * trailWeight * deltaTime)), 1.), 0.);

    // read optical flow
    vec2 opticalFlowForce = imageLoad(optFlowMap, ivec2(coord / opticalFlowDownScale)).xy;
    opticalFlowForce = opticalFlowForce * 2. - 1.;
    float opticalFlowMag = length(opticalFlowForce);
    // float opticalDecayRate = 1. - 0.8 * opticalFlowMag;
    float opticalDecayRate = 1.;
    
    vec4 blurredTrail = min(max((newTrail * opticalDecayRate * decayRate * deltaTime), 0.), 1.);

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

    out_color = blurredTrail;
    out_color.w = 1.; // TODO if we remove this, everything breaks. why? 
}