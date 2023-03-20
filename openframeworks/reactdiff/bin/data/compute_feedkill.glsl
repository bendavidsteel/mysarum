#version 440

layout(rg16,binding=1) uniform restrict image2D feedkillMap;

uniform ivec2 resolution;
uniform float time;
uniform float speed;

float get_third_degree_polynomial_out(float x, vec4 coefs) {
    vec4 xs = vec4(1.);
    xs.y = x;
    xs.z = pow(x, 2);
    xs.w = pow(x, 3);
    return dot(xs, coefs);
}

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(){
    vec2 coord = gl_GlobalInvocationID.xy;
    vec2 uv = coord / resolution;
    
    float kill_min = 0.045;
    float kill_range = 0.025;
    float feed_min = 0.01;
    float feed_range = 0.09;

    float feed = feed_min + (feed_range * uv.y);

    vec4 kill_low_coefs = vec4(0.01412, 1.91897, -25.11451, 100.75403);
    float kill_low = get_third_degree_polynomial_out(feed, kill_low_coefs);
    vec4 kill_high_coefs = vec4(0.04666, 0.93116, -12.58194, 46.69186);
    float kill_high = get_third_degree_polynomial_out(feed, kill_high_coefs);
    kill_low = max(kill_low, kill_min);
    kill_high = min(kill_high, kill_min + kill_range);
    float kill = kill_low + ((kill_high - kill_low) * uv.x);

    vec4 vals = vec4(feed, kill, 0., 0.);

	imageStore(feedkillMap, ivec2(coord), vals);
}