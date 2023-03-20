#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;

out vec4 out_color;

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);

	// float x = float(coord.x);
	// float y = float(coord.y);

	// float theta = atan(y/x);
	// float r = sqrt(pow(x, 2) + pow(y, 2));

	vec2 chems = imageLoad(reactionMap, coord).xy;

	out_color = vec4(0., 0., 0., 1.);
	out_color += chems.x * vec4(colourA, 1.);
	out_color += chems.y * vec4(colourB, 1.);
}