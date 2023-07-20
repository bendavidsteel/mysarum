#version 440

uniform ivec2 resolution;
uniform float time;
uniform float speed;
uniform float feedMin;
uniform float feedRange;
uniform int pattern;

out vec4 out_color;

float get_third_degree_polynomial_out(float x, vec4 coefs) {
    vec4 xs = vec4(1.);
    xs.y = x;
    xs.z = pow(x, 2);
    xs.w = pow(x, 3);
    return dot(xs, coefs);
}

void main(){
    vec2 coord = gl_FragCoord.xy;
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

    out_color = vec4(feed, kill, 1., 1.);
}