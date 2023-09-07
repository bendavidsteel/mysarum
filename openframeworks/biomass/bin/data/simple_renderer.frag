#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

uniform sampler2DRect flowMap;
uniform sampler2DRect reactionMap;
uniform sampler2DRect trailMap;
uniform sampler2DRect optFlowMap;
uniform sampler2DRect audioMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;
uniform float trail_height;
uniform int opticalFlowDownScale;
uniform int display;

out vec4 out_color;

void main()
{
	vec2 coord = gl_FragCoord.xy;

	vec2 audio = texture(audioMap, coord).xy;
	vec3 audioColour = colourC * audio.x;
	audioColour += colourD * audio.y;

	vec3 colour = vec3(0.);

	if (display == 4) {
		vec2 opticalFlowForce = texture(optFlowMap, coord / opticalFlowDownScale).xy;
		opticalFlowForce = (2 * opticalFlowForce) - 1; // convert to -1-1 range
		colour = vec3(opticalFlowForce.x, opticalFlowForce.y, -opticalFlowForce.x);

	} else if (display == 3) {
		colour = audioColour;
	}

    out_color = vec4(colour, 1.);
}