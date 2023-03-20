#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=1) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=2) uniform restrict image3D trailMap;

layout(rgba8,binding=3) uniform restrict image3D flowMap;

uniform ivec2 screen_res;
uniform ivec3 trail_res;

out vec4 out_color;

void main()
{
	vec2 coord = gl_FragCoord.xy / screen_res;
	ivec2 trail_coord = ivec2(coord * trail_res.xy);

	out_color = vec4(0., 0., 0., 1.);
	for (int i = 0; i < trail_res.z; i++)
	{
		vec4 trail_sample = imageLoad(trailMap, ivec3(trail_coord, i));

		out_color += trail_sample.r * allSpecies[0].colour;
		out_color += trail_sample.g * allSpecies[1].colour;
		out_color += trail_sample.b * allSpecies[2].colour;
		out_color += trail_sample.a * allSpecies[3].colour;
	}
}