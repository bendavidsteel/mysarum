#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;
layout(rg16,binding=4) uniform restrict image2D optFlowMap; // TODO switch to readonly
layout(rg16,binding=7) uniform restrict image2D audioMap;

uniform sampler2DRect flowMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform int opticalFlowDownScale;
uniform float reactionFlowMag;
uniform float feedMin;
uniform float feedRange;

float get_third_degree_polynomial_out(float x, vec4 coefs) {
    vec4 xs = vec4(1.);
    xs.y = x;
    xs.z = pow(x, 2);
    xs.w = pow(x, 3);
    return dot(xs, coefs);
}

vec2 get_feedkill(vec2 coord) {
    vec2 uv = coord / vec2(resolution);
    
    float kill_min = 0.045;
    float kill_range = 0.025;

    float feed = feedMin + (feedRange * uv.y);

    vec4 kill_low_coefs = vec4(0.01412, 1.91897, -25.11451, 100.75403);
    float kill_low = get_third_degree_polynomial_out(feed, kill_low_coefs);
    vec4 kill_high_coefs = vec4(0.04666, 0.93116, -12.58194, 46.69186);
    float kill_high = get_third_degree_polynomial_out(feed, kill_high_coefs);
    kill_low = max(kill_low, kill_min);
    kill_high = min(kill_high, kill_min + kill_range);
    float kill = kill_low + ((kill_high - kill_low) * uv.x);

    return vec2(feed, kill);
}

layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // previous values
    vec2 previous = imageLoad(reactionMap, coord).xy;
    float a = previous.x;
    float b = previous.y;

    vec2 audio = imageLoad(audioMap, coord).xy;
    float audioMag = length(audio);

    // get flow map
    vec2 simplex_flow = texture(flowMap, coord).xy;
    simplex_flow = (2 * simplex_flow) - 1; //convert flow to -1 to 1 range

    vec2 optical_flow = imageLoad(optFlowMap, coord / opticalFlowDownScale).xy;
    optical_flow = (2 * optical_flow) - 1; //convert flow to -1 to 1 range
    float opticalFlowMag = length(optical_flow);

    vec2 flow = (reactionFlowMag + 15 * opticalFlowMag + 5 * audioMag) * simplex_flow;
    float flow_left = 0.2 + 0.2 * flow.x;
    float flow_right = 0.2 - 0.2 * flow.x;
    float flow_up = 0.2 + 0.2 * flow.y;
    float flow_down = 0.2 - 0.2 * flow.y;
    float flow_lu = 0.05 + 0.025 * flow.x + 0.025 * flow.y;
    float flow_ru = 0.05 - 0.025 * flow.x + 0.025 * flow.y;
    float flow_ld = 0.05 + 0.025 * flow.x - 0.025 * flow.y;
    float flow_rd = 0.05 - 0.025 * flow.x - 0.025 * flow.y;

    // compute laplacian
    ivec2 neighbourCoord = coord + ivec2(-1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    vec2 neighbours = imageLoad(reactionMap, neighbourCoord).xy * flow_ld;

    neighbourCoord = coord + ivec2(-1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_left;

    neighbourCoord = coord + ivec2(-1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_lu;

    neighbourCoord = coord + ivec2(0, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_down;

    neighbourCoord = coord + ivec2(0, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_up;

    neighbourCoord = coord + ivec2(1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_rd;

    neighbourCoord = coord + ivec2(1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_right;

    neighbourCoord = coord + ivec2(1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += imageLoad(reactionMap, neighbourCoord).xy * flow_ru;

    vec2 laplacian = neighbours - previous;

    float reaction = a * b * b;
    vec2 reactionVec = reaction * vec2(-1, 1);

    vec2 feedkill = get_feedkill(gl_GlobalInvocationID.xy);
    float f = feedkill.x;
    float k = feedkill.y;

    float feed = f * (1 - a);
    float kill = -1 * (k + f) * b;
    vec2 feedkillVec = vec2(feed, kill);

    vec2 diffusion = vec2(1.0, 0.5) + vec2(0.15 * audio.x, 0.3 * audio.y);

    vec2 newValues = previous + (deltaTime * ((diffusion * laplacian) + reactionVec + feedkillVec));

    vec4 vals = vec4(newValues, 0., 1.);
	imageStore(reactionMap, coord, vals);
}