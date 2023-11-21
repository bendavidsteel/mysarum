#version 440

uniform sampler2DRect reactionMap;

uniform vec3 colourA;
uniform vec3 colourB;

in vec2 texCoordVarying;

out vec4 out_color;

void main()
{
	vec2 coord = texCoordVarying;

	vec2 chems = texture(reactionMap, coord).xy;

	vec3 colour = chems.x * colourA;
	colour += chems.y * colourB;

	out_color = vec4(colour, 1.);
	// out_color = vec4(1., 1., 1., 1.);
}