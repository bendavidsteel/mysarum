#version 440

layout(rgba32f,binding=1) uniform restrict image2D trailMap;

uniform ivec2 resolution;
uniform float time;
uniform float deltaTime;
uniform float diffuseRate;
uniform float decayRate;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(){

    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // accumulator
    vec4 originalTrail = imageLoad(trailMap, coord);
    vec4 sum = vec4(0.0);
    
    //blur box size
    const int dim = 1;

    for( int i = -dim; i <= dim; i++ ){
        for( int j = -dim; j <= dim; j++ ){
            ivec2 coord = coord + ivec2(i, j);
            coord = min(resolution, max(coord, 0));
            sum += imageLoad(trailMap, coord);
        }
    }

    vec4 blurredTrail = sum / 9;
    float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
    blurredTrail = (originalTrail * (1 - diffuseWeight)) + (blurredTrail * diffuseWeight);
    
    //DiffusedTrailMap[id.xy] = blurredTrail * saturate(1 - decayRate * deltaTime);
    vec4 newTrail = min(max((blurredTrail * decayRate * deltaTime), 0.), 1.);
	imageStore(trailMap, coord, newTrail);
}