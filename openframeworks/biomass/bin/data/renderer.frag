#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=5) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=6) uniform restrict image3D trailMap;

layout(rgba8,binding=0) uniform restrict image3D flowMap;

layout(rg16,binding=3) uniform restrict image2D reactionMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;
uniform float trail_height;

uniform int trail_depth;

out vec4 out_color;

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);

	vec2 chems = imageLoad(reactionMap, coord).xy;

	vec3 pos = vec3(coord, chems.y);
	float pos_height = chems.y * chem_height;
	float dist_to_light = distance(pos.xy, light.xy);

	vec3 lighting = vec3(1.);
	float falloff = resolution.x / 3;
	vec3 light_falloff = exp(-dist_to_light / (30 * falloff)) + 0.3 * vec3(sin(-dist_to_light / falloff), sin(1 - (dist_to_light / falloff)), sin(2 - (dist_to_light / falloff)));

	float max_dist_to_other_peak = dist_to_light * (chem_height - pos_height) / (light.z - pos_height);
	vec2 dir_to_light = normalize(light.xy - pos.xy);
	
	for (float dist = 0.; dist < max_dist_to_other_peak; dist++)
	{
		vec2 other_peak = pos.xy + (dir_to_light * dist);
		if (other_peak.x < 0 || other_peak.x >= resolution.x || other_peak.y < 0 || other_peak.y >= resolution.y) {
			break;
		}

		float other_peak_height = imageLoad(reactionMap, ivec2(other_peak)).y * chem_height;
		float light_height = pos_height + (dist * (light.z - pos_height) / dist_to_light);
		if (other_peak_height > light_height) {
			// in shadow
			lighting = vec3(0.4);
			break;
		}
	}
	lighting *= light_falloff;

	vec3 colour = chems.x * colourA * lighting;
	colour += chems.y * colourB * lighting;

	for (int i = 0; i < trail_depth; i++)
	{
		float trail_d = dist_to_light * (trail_height + float(i) - pos_height) / (light.z - pos_height);
		vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
		vec4 trail_shadow = imageLoad(trailMap, ivec3(trail_coord, i));

		if (trail_shadow.r > 0.1) {
			colour *= (1 - 0.5 * trail_shadow.r) * (0.5 + 0.5 * allSpecies[0].colour.rgb);
		}
		if (trail_shadow.g > 0.1) {
			colour *= (1 - 0.5 * trail_shadow.g) * (0.5 + 0.5 * allSpecies[1].colour.rgb);
		}
		if (trail_shadow.b > 0.1) {
			colour *= (1 - 0.5 * trail_shadow.b) * (0.5 + 0.5 * allSpecies[2].colour.rgb);
		}
		if (trail_shadow.a > 0.1) {
			colour *= (1 - 0.5 * trail_shadow.a) * (0.5 + 0.5 * allSpecies[3].colour.rgb);
		}
	}

	for (int i = 0; i < trail_depth; i++)
	{
		vec4 trail = imageLoad(trailMap, ivec3(coord, i));

		if (trail.r > 0.1) {
			colour = mix(colour, allSpecies[0].colour.rgb * light_falloff, trail.r);
		}
		if (trail.g > 0.1) {
			colour = mix(colour, allSpecies[1].colour.rgb * light_falloff, trail.g);
		}
		if (trail.b > 0.1) {
			colour = mix(colour, allSpecies[2].colour.rgb * light_falloff, trail.b);
		}
		if (trail.a > 0.1) {
			colour = mix(colour, allSpecies[3].colour.rgb * light_falloff, trail.a);
		}
	}

	out_color = vec4(colour, 1.);
}