#version 440

layout(rgba8,binding=2) uniform restrict image3D trailMap;

uniform ivec3 resolution;
uniform float deltaTime;
uniform float diffuseRate;
uniform ivec3 blurDir;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(){

    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);

    // accumulator
    vec4 originalTrail = imageLoad(trailMap, coord);
    vec4 sum = vec4(0.0);
    
    //blur box size
    const int dim = 1;

    for( int i = -dim; i <= dim; i++ ){
        ivec3 pointCoord = coord + (i * blurDir);
        pointCoord = min(resolution-1, max(pointCoord, 0));
        sum += imageLoad(trailMap, pointCoord);
    }

    vec4 blurredTrail = sum / 3;
    float diffuseWeight = clamp(diffuseRate * deltaTime, 0, 1);
    blurredTrail = (originalTrail * (1 - diffuseWeight)) + (blurredTrail * diffuseWeight);
    
	imageStore(trailMap, coord, blurredTrail);
}