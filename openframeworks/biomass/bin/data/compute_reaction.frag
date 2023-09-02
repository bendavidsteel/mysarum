#version 440

layout(rg16,binding=4) uniform restrict image2D optFlowMap; // TODO switch to readonly
layout(rg16,binding=7) uniform restrict image2D audioMap;

uniform sampler2DRect flowMap;
uniform sampler2DRect lastReactionMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform int opticalFlowDownScale;
uniform float reactionFlowMag;
uniform float feedMin;
uniform float feedRange;
uniform int mapFactor;
uniform int initialise;

out vec4 out_color;

// A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
uint hash( uint x ) {
	x += ( x << 10u );
	x ^= ( x >>  6u );
	x += ( x <<  3u );
	x ^= ( x >> 11u );
	x += ( x << 15u );
	return x;
}

// Compound versions of the hashing algorithm I whipped together.
uint hash( uvec2 v ) { return hash( v.x ^ hash(v.y)                         ); }
uint hash( uvec3 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z)             ); }
uint hash( uvec4 v ) { return hash( v.x ^ hash(v.y) ^ hash(v.z) ^ hash(v.w) ); }

// Construct agent float with half-open range [0:1] using low 23 bits.
// All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
float floatConstruct( uint m ) {
    const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
    const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32

    m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
    m |= ieeeOne;                          // Add fractional part to 1.0

    float  f = uintBitsToFloat( m );       // Range [1:2]
    return f - 1.0;                        // Range [0:1]
}

// Pseudo-random value in half-open range [0:1].
float random( float x ) { return floatConstruct(hash(floatBitsToUint(x))); }
float random( vec2  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec3  v ) { return floatConstruct(hash(floatBitsToUint(v))); }
float random( vec4  v ) { return floatConstruct(hash(floatBitsToUint(v))); }

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

void main(){
    ivec2 coord = ivec2(gl_FragCoord.xy);

    // previous values
    float a, b;
    if (initialise == 1) {
        a = 1.;
        vec2 uv = vec2(coord) / vec2(resolution);
        if (random(uv) < 0.1) {
            b = 1.;
        } else {
            b = 0.;
        }
    } else {
        vec2 previous = texture(lastReactionMap, coord).xy;
        a = previous.x;
        b = previous.y;
    }

    float audio = imageLoad(audioMap, coord).x;

    // get flow map
    vec2 simplex_flow = texture(flowMap, coord).xy;
    simplex_flow = (2 * simplex_flow) - 1; //convert flow to -1 to 1 range

    vec2 optical_flow = imageLoad(optFlowMap, coord / opticalFlowDownScale).xy;
    optical_flow = (2 * optical_flow) - 1; //convert flow to -1 to 1 range
    float opticalFlowMag = length(optical_flow);

    vec2 flow = 0. * (reactionFlowMag + 15 * opticalFlowMag + 5 * audio) * simplex_flow;
    float flow_left = 0.2 + 0.2 * flow.x;
    float flow_right = 0.2 - 0.2 * flow.x;
    float flow_up = 0.2 + 0.2 * flow.y;
    float flow_down = 0.2 - 0.2 * flow.y;
    float flow_lu = 0.05 + 0.025 * flow.x + 0.025 * flow.y;
    float flow_ru = 0.05 - 0.025 * flow.x + 0.025 * flow.y;
    float flow_ld = 0.05 + 0.025 * flow.x - 0.025 * flow.y;
    float flow_rd = 0.05 - 0.025 * flow.x - 0.025 * flow.y;

    // compute laplacian
    int kernelSize = 1;
    ivec2 neighbourCoord = coord + kernelSize * ivec2(-1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    vec2 neighbours = texture(lastReactionMap, neighbourCoord).xy * flow_ld;

    neighbourCoord = coord + kernelSize * ivec2(-1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_left;

    neighbourCoord = coord + kernelSize * ivec2(-1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_lu;

    neighbourCoord = coord + kernelSize * ivec2(0, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_down;

    neighbourCoord = coord + kernelSize * ivec2(0, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_up;

    neighbourCoord = coord + kernelSize * ivec2(1, -1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_rd;

    neighbourCoord = coord + kernelSize * ivec2(1, 0);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_right;

    neighbourCoord = coord + kernelSize * ivec2(1, 1);
    neighbourCoord = min(resolution-1, max(neighbourCoord, 0));
    neighbours += texture(lastReactionMap, neighbourCoord).xy * flow_ru;

    if (initialise == 1) {
        neighbours = vec2(a, b);
    }
    vec2 laplacian = neighbours - vec2(a, b);

    float reaction = a * b * b;
    vec2 reactionVec = reaction * vec2(-1, 1);

    vec2 uv = vec2(coord) / vec2(resolution);
    vec2 centre = vec2(0.5, 0.5);
    vec2 from_centre = uv - centre;
    float r = length(from_centre);

    vec2 feedkill;
    // if (r > 0.2) {
    //     feedkill = vec2(0.096, 0.057);
    // } else {
    //     feedkill = vec2(0.015, 0.045);
    // }

    feedkill = get_feedkill(gl_FragCoord.xy);
    
    // feedkill = vec2(0.096, 0.057);
    // feedkill = vec2(0.073, 0.061);
    // feedkill = vec2(0.035, 0.059);
    // feedkill = vec2(0.021, 0.053);
    // feedkill = vec2(0.015, 0.045);

    float f = feedkill.x;
    float k = feedkill.y;

    float feed = f * (1 - a);
    float kill = -1 * (k + f) * b;
    vec2 feedkillVec = vec2(feed, kill);

    vec2 diffusion = vec2(1.0, 0.5) + (0.5 * audio * vec2(0.15, 0.3));

    vec2 newValues = vec2(a, b) + (deltaTime) * ((diffusion * laplacian) + reactionVec + feedkillVec);

    out_color = vec4(newValues, 0., 1.);
}