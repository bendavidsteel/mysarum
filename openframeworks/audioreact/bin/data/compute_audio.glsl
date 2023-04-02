#version 440

#define PI 3.14159265359

struct SpectralComponent{
    vec4 value;
};

layout(rgba16,binding=0) uniform restrict image2D audioMap;

layout(std140, binding=1) buffer spectrum{
    SpectralComponent allSpectrum[];
};

uniform ivec2 resolution;
uniform float deltaTime;
uniform int numBands;
uniform float angle;

float map(float val, float a, float b, float c, float d) {
    float normVal = (val - a) / (b - a);
    return c + (normVal * d);
}

float easeIn(float x) {
    return pow(x, 2);
}

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(1.0) * vec2(coord) / vec2(resolution);
    vec2 centre = vec2(0.5, 0.5);
    vec2 from_centre = uv - centre;
    float this_angle = atan(from_centre.y, from_centre.x);
    float alt_angle = mod(this_angle + 2 * PI, 2 * PI) - PI;

    int type = 0;

    vec4 vals = vec4(0., 0., 0., 1.);
    if (type == 0) {
        vals = imageLoad(audioMap, coord).xyzw;
        vals.zw = vec2(0., 1.);

        float scale = easeIn(length(from_centre) * 2);
        // float scale = length(from_centre) * 2;
        float index = scale * numBands;
        if (index >= numBands) {
            index = numBands - 1 - (index - numBands);
        }

        float lowerSpectrumVal = allSpectrum[int(floor(index))].value.x;
        float upperSpectrumVal = allSpectrum[int(ceil(index))].value.x;
        float spectrumVal = mix(lowerSpectrumVal, upperSpectrumVal, fract(index));
        // float spectrumMapped = map(spectrumVal, -12., 1., 0., 1.);
        float spectrumMapped = exp((spectrumVal - 1) / 4);
        vals.x = spectrumMapped;
    } else if (type == 1) {
        if (coord.x == resolution.x - 1) {
            int index = int((1 - uv.y) * numBands);
            float spectrumVal = allSpectrum[index].value.x;
            float spectrumMapped = exp((spectrumVal - 1) / 4);
            vals.xyz = vec3(spectrumMapped);
        } else {
            vals.xyz = imageLoad(audioMap, coord + ivec2(1, 0)).xyz;
        }
    }

    // neural network idea
    // vec4 layer1 = vec4(0.);
    // layer1.x = dot(uv, vec2(-0.2, 0.1));
    // layer1.y = dot(uv, vec2(0.3, -0.57));
    // layer1.z = dot(uv, vec2(-0.22, 0.17));
    // layer1.w = dot(uv, vec2(0.67, 0.11));

    // layer1 = max(layer1, 0.);

    // vec3 layer2 = vec3(0.);
    // layer2.x = dot(layer1, vec4(0.1, 0.2, 0.3, 0.4));
    // layer2.y = dot(layer1, vec4(0.5, 0.6, 0.7, 0.8));
    // layer2.z = dot(layer1, vec4(0.9, 0.1, 0.2, 0.3));
    
    
	imageStore(audioMap, coord, vals);
}