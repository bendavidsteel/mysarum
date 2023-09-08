#version 440

uniform float time;
uniform float bass;
uniform ivec2 mapSize;
uniform ivec2 resolution;
uniform sampler2DRect tex;
uniform sampler2DRect mask;
uniform float bps;

out vec4 out_color;

#define SHOW_NOISE 0
#define SRGB 0
// 0: Addition, 1: Screen, 2: Overlay, 3: Soft Light, 4: Lighten-Only
#define BLEND_MODE 3
#define SPEED 2.0
#define INTENSITY 0.075
// What gray level noise should tend to.
#define MEAN 0.0
// Controls the contrast/variance of noise.
#define VARIANCE 0.5

vec3 channel_mix(vec3 a, vec3 b, vec3 w) {
    return vec3(mix(a.r, b.r, w.r), mix(a.g, b.g, w.g), mix(a.b, b.b, w.b));
}

float gaussian(float z, float u, float o) {
    return (1.0 / (o * sqrt(2.0 * 3.1415))) * exp(-(((z - u) * (z - u)) / (2.0 * (o * o))));
}

vec3 madd(vec3 a, vec3 b, float w) {
    return a + a * b * w;
}

vec3 screen(vec3 a, vec3 b, float w) {
    return mix(a, vec3(1.0) - (vec3(1.0) - a) * (vec3(1.0) - b), w);
}

vec3 overlay(vec3 a, vec3 b, float w) {
    return mix(a, channel_mix(
        2.0 * a * b,
        vec3(1.0) - 2.0 * (vec3(1.0) - a) * (vec3(1.0) - b),
        step(vec3(0.5), a)
    ), w);
}

vec3 soft_light(vec3 a, vec3 b, float w) {
    return mix(a, pow(a, pow(vec3(2.0), 2.0 * (vec3(0.5) - b))), w);
}

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

mat3 get_colour_rotation(int theta)
{
    theta = theta % 6;
    if (theta == 0)
    {
        return mat3(1., 0., 0., 0., 1., 0., 0., 0., 1.);
    }
    else if (theta == 1)
    {
        return mat3(0.5, 0.5, 0., 0., 0.5, 0.5, 0.5, 0., 0.5);
    }
    else if (theta == 2)
    {
        return mat3(0., 1., 0., 0., 0., 1., 1., 0., 0.);
    }
    else if (theta == 3)
    {
        return mat3(0., 0.5, 0.5, 0.5, 0., 0.5, 0.5, 0.5, 0.);
    }
    else if (theta == 4)
    {
        return mat3(0., 0., 1., 1., 0., 0., 0., 1., 0.);
    }
    else if (theta == 5)
    {
        return mat3(0.5, 0., 0.5, 0.5, 0.5, 0., 0., 0.5, 0.5);
    }
}

void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 uv = coord / resolution.xy;

    // do mapping
    // float left = 0.;
    // float right = 0.6;
    // float up = 1.0;
    // float down = 0.4;

    // vec2 mapped_uv;
    // mapped_uv.x = (uv.x - left) / (right - left);
    // mapped_uv.y = uv.y;

    // vec2 mapped_coord = mapped_uv.xy * resolution.xy;

    // color = texture(tex, mapped_coord);

    // vec2 mapped_uv = uv;

    vec2 centre = vec2(0.5);
    vec2 from_centre = uv - centre;

    float theta = atan(from_centre.y, from_centre.x);
    float r = length(from_centre);

    // r *= 0.9 + 0.1 * pow(bass, 2.);
    // theta += 0.01 * random(r);

    // uv = centre + vec2(cos(theta), sin(theta)) * r;

    // uv += 0.1 * vec2(sin(0.25 * bps * time), cos(0.25 * bps * time));

    vec2 mapped_uv = uv;

    mapped_uv /= 1.0;

    float speed = 0.05;
    // mapped_uv += 0.25 + 0.25 * vec2(sin(speed * time), cos(speed * time));

    
    // centre = vec2(0.5);// + 0.3 * vec2(sin(speed * time), cos(speed * time));
    // from_centre = uv - centre;
    // r = pow(pow(from_centre.x, 2.0) + (pow(from_centre.y, 2.0) / pow(1.5, 2)), 0.5);
    // if (r < bass / 2.) {
    //     float mapped_r = pow(r, 0.5);
    //     mapped_uv = centre + from_centre * mapped_r;
    // }

    ivec2 sampled_coord = ivec2(mapped_uv.xy * mapSize.xy);
    vec4 color = vec4(0.);
    color.a = 1.0;
    float sharpness = 1.;
    color.rgb += sharpness * texture(tex, sampled_coord + ivec2(0, 0)).rgb;
    color.rgb += ((1 - sharpness) / 4.) * texture(tex, sampled_coord + ivec2(1, 0)).rgb;
    color.rgb += ((1 - sharpness) / 4.) * texture(tex, sampled_coord + ivec2(-1, 0)).rgb;
    color.rgb += ((1 - sharpness) / 4.) * texture(tex, sampled_coord + ivec2(0, 1)).rgb;
    color.rgb += ((1 - sharpness) / 4.) * texture(tex, sampled_coord + ivec2(0, -1)).rgb;

    #if SRGB
    color = pow(color, vec4(2.2));
    #endif
    
    float t = time * float(SPEED);
    float seed = dot(uv, vec2(12.9898, 78.233));
    float noise = fract(sin(seed) * 43758.5453 + t);
    noise = gaussian(noise, float(MEAN), float(VARIANCE) * float(VARIANCE));
    
    #if SHOW_NOISE
    color = vec4(noise);
    #else    
    // Ignore these mouse stuff if you're porting this
    // and just use an arbitrary intensity value.
    float w = float(INTENSITY);
	
    vec3 grain = vec3(noise) * (1.0 - color.rgb);
    
    #if BLEND_MODE == 0
    color.rgb += grain * w;
    #elif BLEND_MODE == 1
    color.rgb = screen(color.rgb, grain, w);
    #elif BLEND_MODE == 2
    color.rgb = overlay(color.rgb, grain, w);
    #elif BLEND_MODE == 3
    color.rgb = soft_light(color.rgb, grain, w);
    #elif BLEND_MODE == 4
    color.rgb = max(color.rgb, grain * w);
    #endif
        
    #if SRGB
    color = pow(color, vec4(1.0 / 2.2));
    #endif
    #endif

    // color.rgb *= get_colour_rotation(int(3. + 3. * sin(bps * time)));

    // uv = coord / resolution.xy;
    // uv += 0.25 * vec2(sin(0.25 * bps * time), cos(0.25 * bps * time));
    // coord = uv * resolution.xy;
    // vec4 colour_artificer = 0.5 * texture(artificer, coord).rgba;
    // colour_artificer += 0.125 * texture(artificer, coord + ivec2(0, 1)).rgba;
    // colour_artificer += 0.125 * texture(artificer, coord + ivec2(0, -1)).rgba;
    // colour_artificer += 0.125 * texture(artificer, coord + ivec2(1, 0)).rgba;
    // colour_artificer += 0.125 * texture(artificer, coord + ivec2(-1, 0)).rgba;

    // if (colour_artificer.a > 0.1) {
    //     color.rgb = vec3(random(time + 2), random(time + 1), random(time));
    //     // color.rgb = 1 - color.rgb;
    // }

    // invert
    // color.rgb = vec3(1.0) - color.rgb;

    out_color = texture(tex, gl_FragCoord.xy);
}
