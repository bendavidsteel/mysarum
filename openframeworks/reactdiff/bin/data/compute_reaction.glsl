#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;
layout(rg16,binding=1) uniform restrict image2D feedkillMap;
layout(rgba8,binding=2) uniform restrict image2D flowMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform vec2 diffusion;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // previous values
    vec2 previous = imageLoad(reactionMap, coord).xy;
    float a = previous.x;
    float b = previous.y;

    // get flow map
    vec2 flow = imageLoad(flowMap, coord).xy;
    flow = (2 * flow) - 1; //convert flow to -1 to 1 range
    float float_strength = 1;
    float flow_left = 0.2 + 0.2 * flow.x;
    float flow_right = 0.2 - 0.2 * flow.x;
    float flow_up = 0.2 + 0.2 * flow.y;
    float flow_down = 0.2 - 0.2 * flow.y;
    float flow_lu = 0.05; // TODO

    // compute laplacian
    ivec2 neighbourCoord = coord + ivec2(-1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    vec2 neighbours = imageLoad(reactionMap, neighbourCoord).xy * 0.05;

    neighbourCoord = coord + ivec2(-1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_left;

    neighbourCoord = coord + ivec2(-1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * 0.05;

    neighbourCoord = coord + ivec2(0, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_down;

    neighbourCoord = coord + ivec2(0, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_up;

    neighbourCoord = coord + ivec2(1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * 0.05;

    neighbourCoord = coord + ivec2(1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_right;

    neighbourCoord = coord + ivec2(1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * 0.05;

    vec2 laplacian = neighbours - previous;

    float reaction = a * b * b;
    vec2 reactionVec = reaction * vec2(-1, 1);

    vec2 feedkill = imageLoad(feedkillMap, coord).xy;
    float f = feedkill.x;
    float k = feedkill.y;

    float feed = f * (1 - a);
    float kill = -1 * (k + f) * b;
    vec2 feedkillVec = vec2(feed, kill);

    vec2 newValues = previous + (deltaTime * ((diffusion * laplacian) + reactionVec + feedkillVec));
    vec4 vals = vec4(newValues, 0., 1.);
    // vec4 vals = vec4(1., 1., 0., 0.);
	imageStore(reactionMap, coord, vals);
}