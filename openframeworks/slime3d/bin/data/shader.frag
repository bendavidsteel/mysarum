#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(rgba8,binding=1) uniform restrict image2D trailMap;

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

out vec4 outputColor;

void main(){
	//this is the fragment shader
	//this is where the pixel level drawing happens
	//gl_FragCoord gives us the x and y of the current pixel its drawing
	
	//we grab the x and y and store them in an int
    ivec2 coord = ivec2(gl_FragCoord.xy);

    vec4 trail = imageLoad(trailMap, coord);
	outputColor = vec4(0.0, 0.0, 0.0, 1.0);
    outputColor += trail.r * allSpecies[0].colour;
    outputColor += trail.g * allSpecies[1].colour;
    outputColor += trail.b * allSpecies[2].colour;
    outputColor += trail.a * allSpecies[3].colour;
}