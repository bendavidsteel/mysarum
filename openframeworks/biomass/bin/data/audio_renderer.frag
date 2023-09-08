#version 440

uniform sampler2DRect audioMap;

uniform vec3 colourC;
uniform vec3 colourD;

in vec2 texCoordVarying;

out vec4 out_color;

void main()
{
	vec2 coord = texCoordVarying;

	vec2 audio = texture(audioMap, coord).xy;
	vec3 colour = colourC * audio.x;
	colour += colourD * audio.y;
	
    out_color = vec4(colour, 1.);
}