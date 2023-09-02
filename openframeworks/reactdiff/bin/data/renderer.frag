#version 440

layout(rg16,binding=0) uniform restrict image2D reactionMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;
uniform vec3 light;
uniform float height;

out vec4 out_color;

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);

	vec2 uv = vec2(coord) / vec2(resolution);

	// raytracing
	vec3 camera = vec3(resolution.x / 2, resolution.y / 2, 100.);
	vec2 aperture = vec2(resolution.x / 2, resolution.y / 2);

	vec2 from_centre = uv - vec2(0.5);
	vec3 ray_origin = camera + vec3(from_centre * aperture, 0.);
	vec3 ray_dir = vec3(0., 0., -1.);

	// TODO the rest

	vec2 chems = imageLoad(reactionMap, coord).xy;

	// calc shadows
	float this_chem_height = 1.;

	vec3 pos = vec3(coord, chems.y);
	float pos_height = chems.y * this_chem_height;
	float dist_to_light = distance(pos.xy, light.xy);

	vec3 lighting = vec3(1.);

	float falloff = resolution.x / 3;
	vec3 light_falloff = vec3(exp(-dist_to_light / (100 * falloff)));

	float height_to_light = light.z - pos_height;
	float max_dist_to_other_peak = dist_to_light * (this_chem_height - pos_height) / height_to_light;
	vec2 dir_to_light = normalize(light.xy - pos.xy);
	
	for (float dist = 0.; dist < max_dist_to_other_peak; dist++)
	{
		vec2 other_peak = pos.xy + (dir_to_light * dist);
		if (other_peak.x < 0 || other_peak.x >= resolution.x || other_peak.y < 0 || other_peak.y >= resolution.y) {
			break;
		}

		float other_peak_height = imageLoad(reactionMap, ivec2(other_peak)).y * this_chem_height;
		float light_height = pos_height + (dist * height_to_light / dist_to_light);
		if (other_peak_height > light_height) {
			// in shadow
			lighting *= vec3(0.4);
			break;
		}
	}
	// lighting *= exp(-dist_to_light / 1000.);

	vec3 colour = chems.x * colourA * lighting;
	colour += chems.y * colourB * lighting;
	out_color = vec4(colour, 1.);
}