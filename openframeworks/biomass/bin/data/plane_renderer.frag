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
uniform sampler2DRect audioMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chemHeight;
uniform float trailHeight;
uniform float time;
uniform float bps;

in vec2 texCoordVarying;

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

float easeOut(float x) {
    float factor = 10.;
    return 1.0 - pow(1.0 - x, factor);
}

float getPosHeight(vec2 pos)
{
	float reactionDisplace = texture(reactionMap, pos).y;
    vec2 trail = texture(trailMap, pos).xy;
	float trailDisplace = trail.x + trail.y;
	vec2 audio = texture(audioMap, pos).xy;
	float audioMag = length(audio);
	float modHeight = ((easeOut(reactionDisplace) * chemHeight) + (trailDisplace * trailHeight)) * (1 + 2 * audioMag);
	return modHeight;
}

float getMaxHeight()
{
	return (chemHeight + 2 * trailHeight) * 3;
}

void main()
{
	vec2 coord = texCoordVarying;

	vec2 chems = texture(reactionMap, coord).xy;
	vec2 audio = texture(audioMap, coord).xy;

	float pos_height = getPosHeight(coord); 
	float max_height = getMaxHeight();

	vec3 pos = vec3(coord, chems.y);
	float dist_to_light = distance(pos.xy, light.xy);

	vec3 lighting = vec3(1.);

	float falloff = resolution.x / 3;
	vec3 light_falloff = vec3(exp(-dist_to_light / (100 * falloff)));

	float height_to_light = light.z - pos_height;
	float max_dist_to_other_peak = dist_to_light * (max_height - pos_height) / height_to_light;
	vec2 dir_to_light = normalize(light.xy - pos.xy);
	
	// for (float dist = 0.; dist < max_dist_to_other_peak; dist += 5.)
	// {
	// 	vec2 other_peak = pos.xy + (dir_to_light * dist);
	// 	if (other_peak.x < 0 || other_peak.x >= resolution.x || other_peak.y < 0 || other_peak.y >= resolution.y) {
	// 		break;
	// 	}

	// 	float other_peak_height = getPosHeight(ivec2(other_peak));
	// 	float light_height = pos_height + (dist * height_to_light / dist_to_light);
	// 	if (other_peak_height > light_height) {
	// 		// in shadow
	// 		lighting *= vec3(0.4);
	// 		break;
	// 	}
	// }
	lighting *= light_falloff;

	vec3 colour = chems.x * colourA * lighting;
	colour += chems.y * colourB * lighting;

	// float trail_d = dist_to_light * (trail_height - pos_height) / height_to_light;
	// vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
	// vec4 trail_shadow = texture(trailMap, trail_coord);

	// if (trail_shadow.r > 0.1) {
	// 	colour *= (1 - 0.5 * trail_shadow.r) * (0.5 + 0.5 * allSpecies[0].colour.rgb);
	// }
	// if (trail_shadow.g > 0.1) {
	// 	colour *= (1 - 0.5 * trail_shadow.g) * (0.5 + 0.5 * allSpecies[1].colour.rgb);
	// }
	// if (trail_shadow.b > 0.1) {
	// 	colour *= (1 - 0.5 * trail_shadow.b) * (0.5 + 0.5 * allSpecies[2].colour.rgb);
	// }
	// if (trail_shadow.a > 0.1) {
	// 	colour *= (1 - 0.5 * trail_shadow.a) * (0.5 + 0.5 * allSpecies[3].colour.rgb);
	// }

	vec4 trail = texture(trailMap, coord);
	// colour.rg = trail.rg;

	vec3 speciesAColour = allSpecies[0].colour.rgb;
	vec3 speciesBColour = allSpecies[1].colour.rgb;
	vec3 speciesCColour = allSpecies[2].colour.rgb;
	vec3 speciesDColour = allSpecies[3].colour.rgb;

	// speciesAColour = vec3(1., 0., 0.);
	// speciesBColour = vec3(0., 1., 0.);
	// speciesCColour = vec3(0., 0., 1.);
	// speciesDColour = vec3(1., 1., 0.);

	if (trail.r > 0.1) {
		colour = mix(colour, speciesAColour * light_falloff, trail.r);
	}
	if (trail.g > 0.1) {
		colour = mix(colour, speciesBColour * light_falloff, trail.g);
	}
	if (trail.b > 0.1) {
		colour = mix(colour, speciesCColour * light_falloff, trail.b);
	}
	// if (trail.a > 0.1) {
	// 	colour = mix(colour, speciesDColour * light_falloff, trail.a);
	// }

	// colour.rg = trail.rg;

	// mat3 colour_rotation = get_colour_rotation(int(6 * sin(time * bps)));//int(audio.x * 6.));
	// colour = mix(colour, colour * colour_rotation, 1.0 * audio.x);
	out_color = vec4(colour, 1.);
}