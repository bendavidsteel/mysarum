#version 440

out vec4 out_color;

uniform sampler2DRect trailMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform float diffuseRate;
uniform ivec2 blurDir;

void main(){

    ivec2 coord = ivec2(gl_FragCoord.xy);

    // accumulator
    vec4 originalTrail = texture(trailMap, coord);
    
    //blur box size
    const int dim = 1;

    vec4 lastTrail = vec4(0.);
    ivec2 pointCoord = coord - blurDir;
    pointCoord = min(resolution-1, max(pointCoord, 0));
    lastTrail = texture(trailMap, pointCoord);

    vec4 nextTrail = vec4(0.);
    pointCoord = coord + blurDir;
    pointCoord = min(resolution-1, max(pointCoord, 0));
    nextTrail = texture(trailMap, pointCoord);

    vec4 sum = lastTrail + originalTrail + nextTrail;

    vec4 blurredTrail = sum / 3;
    float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
    blurredTrail = mix(originalTrail, blurredTrail, diffuseWeight);
    
    out_color = blurredTrail;
    out_color.a = 1;
}