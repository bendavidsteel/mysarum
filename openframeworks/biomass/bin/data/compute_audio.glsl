#version 440

#define PI 3.14159265359

const int maxPoints = 5;

struct Component{
    vec4 value;
};

layout(std140, binding=5) buffer spectrum{
    Component allSpectrum[];
};

layout(std140, binding=6) buffer points{
    Component allPoints[];
};

layout(rg16,binding=7) uniform restrict image2D audioMap;

uniform ivec2 resolution;
uniform float deltaTime;
uniform int numBands;
uniform float angle;
uniform float rms;

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

float getSpectrum(float cassini, float dist) {
    float scale = pow(pow(cassini, 1. / (2. * dist)), 2.) / rms;
    // float scale = length(from_centre) * 2;
    float index = scale * numBands;
    if (ceil(index) >= numBands) {
        index = numBands - 1 - (index - numBands);
    }

    float lowerSpectrumVal = allSpectrum[max(int(floor(index)), 0)].value.x;
    float upperSpectrumVal = allSpectrum[min(int(ceil(index)), numBands-1)].value.x;
    float spectrumVal = mix(lowerSpectrumVal, upperSpectrumVal, pow(fract(index), 2.));
    // float spectrumMapped = exp((spectrumVal - 1) / 4);
    return spectrumVal;
}

layout(local_size_x = 20, local_size_y = 20, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(1.0) * vec2(coord) / vec2(resolution);
    vec2 centre = vec2(0.5, 0.5);
    vec2 from_centre = uv - centre;

    float cassini = 1;
    float a = 0;
    for (int i = 0; i < maxPoints; i++) {
        vec2 pos = allPoints[i].value.xy;
        float strength = allPoints[i].value.z;
        a += strength;
        cassini *= mix(1., sum(pow(from_centre - pos, vec2(2.))), strength);
    }
    float spec = getSpectrum(cassini, a);

    imageStore(audioMap, coord, vec4(spec, 0., 0., 1.));
}