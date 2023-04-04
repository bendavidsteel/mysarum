#version 440

#define PI 3.14159265359

struct SpectralComponent{
    vec4 value;
};

layout(rgba16,binding=0) uniform restrict image2D audioMap;
layout(rgba8,binding=1) uniform restrict image2D flowMap;

layout(std140, binding=1) buffer spectrum{
    SpectralComponent allSpectrum[];
};

uniform ivec2 resolution;
uniform float deltaTime;
uniform int numBands;
uniform float angle;
uniform float rms;
uniform float dissonance;
uniform float numPoints;

float map(float val, float a, float b, float c, float d) {
    float normVal = (val - a) / (b - a);
    return c + (normVal * d);
}

float easeIn(float x) {
    return pow(x, 2);
}

float sum(vec2 v) {
    return dot(v, vec2(1., 1.));
}

float getSpectrum(float cassini) {
    float scale = sqrt(cassini) / rms;
    // float scale = length(from_centre) * 2;
    float index = scale * numBands;
    if (ceil(index) >= numBands) {
        index = numBands - 1 - (index - numBands);
    }

    float lowerSpectrumVal = allSpectrum[max(int(floor(index)), 0)].value.x;
    float upperSpectrumVal = allSpectrum[min(int(ceil(index)), numBands-1)].value.x;
    float spectrumVal = mix(lowerSpectrumVal, upperSpectrumVal, pow(fract(index), 2.));
    float spectrumMapped = exp((spectrumVal - 1) / 4);
    return spectrumMapped;
}

layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(1.0) * vec2(coord) / vec2(resolution);
    vec2 centre = vec2(0.5, 0.5);
    vec2 from_centre = uv - centre;
    float this_angle = atan(from_centre.y, from_centre.x);
    float this_r = length(from_centre);

    float a = pow(rms, 5) / 2;
    float cassini;
    if (this_angle >= 0) {
        float frac = abs(this_angle - PI) / PI;
        float a_cassini = pow(this_r, 4.) - 2 * pow(a, 2.) * pow(this_r, 2.) * cos(numPoints * (angle + this_angle)) + pow(a, 4.);
        float b_cassini = pow(this_r, 4.) - 2 * pow(a, 2.) * pow(this_r, 2.) * cos(numPoints * -PI) + pow(a, 4.);
        cassini = mix(b_cassini, a_cassini, pow(1 - frac, 3.));
    } else if (this_angle < 0) {
        float frac = abs(this_angle + PI) / PI;
        float a_cassini = pow(this_r, 4.) - 2 * pow(a, 2.) * pow(this_r, 2.) * cos(numPoints * (angle + this_angle)) + pow(a, 4.);
        float b_cassini = pow(this_r, 4.) - 2 * pow(a, 2.) * pow(this_r, 2.) * cos(numPoints * PI) + pow(a, 4.);
        cassini = mix(b_cassini, a_cassini, pow(1 - frac, 3.));
    }

    vec4 vals = vec4(0., 0., 0., 1.);
    vals = imageLoad(audioMap, coord).xyzw;
    vals.zw = vec2(0., 1.);

    vals.x = getSpectrum(cassini);

	imageStore(audioMap, coord, vals);
}