#version 440

uniform sampler2DRect optFlowMap;
uniform sampler2DRect audioMap;

uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform int opticalFlowDownScale;
uniform int display;
uniform ivec2 screen_res;

out vec4 out_color;

void main()
{
	vec2 coord = gl_FragCoord.xy * resolution / screen_res;

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