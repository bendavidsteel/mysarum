#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;
layout(rg16,binding=1) uniform restrict image2D trailMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;
uniform float trail_height;

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
	// vec3 light_falloff = vec3(exp(-dist_to_light / falloff));
	vec3 light_falloff = 0.7 + 0.3 * vec3(sin(-dist_to_light / falloff), sin(1 - (dist_to_light / falloff)), sin(2 - (dist_to_light / falloff)));

	float max_dist_to_other_peak = dist_to_light * (chem_height - pos_height) / (light.z - pos_height);
	vec2 dir_to_light = normalize(light.xy - pos.xy);

	for (float d = 0.; d < max_dist_to_other_peak; d += 1.)
	{
		vec2 other_peak = pos.xy + (dir_to_light * d);
		if (other_peak.x < 0. || other_peak.x >= resolution.x || other_peak.y < 0. || other_peak.y >= resolution.y) {
			break;
		}

		float other_peak_height = imageLoad(reactionMap, ivec2(other_peak)).y * chem_height;
		float light_height = pos_height + (d * (light.z - pos_height) / dist_to_light);
		if (other_peak_height > light_height) {
			// in shadow
			lighting = vec3(0.4);
			break;
		}
	}
	lighting *= light_falloff;

	vec3 colour = chems.x * colourA * lighting;
	colour += chems.y * colourB * lighting;

	float trail_d = dist_to_light * (trail_height - pos_height) / (light.z - pos_height);
	vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
	float trail_shadow = imageLoad(trailMap, ivec2(trail_coord)).x;
	float trail = imageLoad(trailMap, coord).x;

	if (trail > 0.) {
		colour += trail * vec3(0., 0., 1.) * light_falloff;
	} else if (trail_shadow > 0.) {
		colour += trail_shadow * vec3(0., 0., 0.3) * light_falloff;
	}

	out_color = vec4(colour, 1.);
}