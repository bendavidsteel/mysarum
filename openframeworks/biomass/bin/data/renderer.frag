#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=3) uniform restrict image2D trailMap;
layout(rg16,binding=0) uniform restrict image2D reactionMap;
layout(rg16,binding=4) uniform restrict image2D optFlowMap;
layout(rg16,binding=7) uniform restrict image2D audioMap;

uniform sampler2DRect flowMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;
uniform float trail_height;
uniform float time;
uniform float bps;

out vec4 out_color;

mat3 get_colour_rotation(int theta)
{
    theta = theta % 6;
    if (theta == 0)
    {
        return mat3(1., 0., 0., 0., 1., 0., 0., 0., 1.);
    }
    else if (theta == 1)
    {
        return mat3(0.5, 0.5, 0., 0., 0.5, 0.5, 0.5, 0., 0.5);
    }
    else if (theta == 2)
    {
        return mat3(0., 1., 0., 0., 0., 1., 1., 0., 0.);
    }
    else if (theta == 3)
    {
        return mat3(0., 0.5, 0.5, 0.5, 0., 0.5, 0.5, 0.5, 0.);
    }
    else if (theta == 4)
    {
        return mat3(0., 0., 1., 1., 0., 0., 0., 1., 0.);
    }
    else if (theta == 5)
    {
        return mat3(0.5, 0., 0.5, 0.5, 0.5, 0., 0., 0.5, 0.5);
    }
}

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);

	vec2 chems = imageLoad(reactionMap, coord).xy;
	vec2 audio = imageLoad(audioMap, coord).xy;

	vec3 audioColour = colourC * audio.x;
	audioColour += colourD * audio.y;

	float this_chem_height = chem_height * (1 + 2 * length(audio));

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
	lighting *= light_falloff;

	vec3 colour = chems.x * colourA * lighting;
	colour += chems.y * colourB * lighting;

	float trail_d = dist_to_light * (trail_height - pos_height) / height_to_light;
	vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
	vec4 trail_shadow = imageLoad(trailMap, ivec2(trail_coord));

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

	vec4 trail = imageLoad(trailMap, coord);

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

	mat3 colour_rotation = get_colour_rotation(int(6 * sin(time * bps)));//int(audio.x * 6.));
	colour = mix(colour, colour * colour_rotation, 1.0 * audio.x);
	out_color = vec4(colour, 1.);
}