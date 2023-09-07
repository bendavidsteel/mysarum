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
uniform sampler2DRect trailMap;
// uniform sampler2DRect audioMap;

uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform vec3 light;
uniform float trail_height;

out vec4 out_color;

void main()
{
	// vec2 coord = gl_FragCoord.xy;
	// vec2 pos = vec2(coord);

	// float dist_to_light = distance(pos.xy, light.xy);
	// vec2 dir_to_light = normalize(light.xy - pos.xy);

	// vec2 audio = texture(audioMap, coord).xy;
	// vec3 audioColour = colourC * audio.x;
	// audioColour += colourD * audio.y;
	// vec3 lighting = vec3(1.);
	// float falloff = resolution.x / 3;
	// vec3 light_falloff = vec3(exp(-dist_to_light / (50 * falloff))) + 0.8 * audioColour;

	vec3 colour = vec3(0.);

    // float pos_height = 0.;
    // float height_to_light = light.z - pos_height;
    // float trail_d = dist_to_light * (trail_height - pos_height) / height_to_light;
    // vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
    // vec4 trail_shadow = texture(trailMap, trail_coord);

    vec3 flow = texture(flowMap, gl_FragCoord.xy).xyz;
    colour = 0.2 * flow;

    // if (trail_shadow.r > 0.1) {
    //     colour *= (1 - 0.5 * trail_shadow.r) * (0.5 + 0.5 * allSpecies[0].colour.rgb);
    // }
    // if (trail_shadow.g > 0.1) {
    //     colour *= (1 - 0.5 * trail_shadow.g) * (0.5 + 0.5 * allSpecies[1].colour.rgb);
    // }
    // if (trail_shadow.b > 0.1) {
    //     colour *= (1 - 0.5 * trail_shadow.b) * (0.5 + 0.5 * allSpecies[2].colour.rgb);
    // }
    // if (trail_shadow.a > 0.1) {
    //     colour *= (1 - 0.5 * trail_shadow.a) * (0.5 + 0.5 * allSpecies[3].colour.rgb);
    // }

    vec4 trail = texture(trailMap, gl_FragCoord.xy);

    vec3 speciesAColour = allSpecies[0].colour.rgb;
	vec3 speciesBColour = allSpecies[1].colour.rgb;
	vec3 speciesCColour = allSpecies[2].colour.rgb;
	vec3 speciesDColour = allSpecies[3].colour.rgb;

	// vec3 speciesAColour = vec3(1., 0., 0.);
	// vec3 speciesBColour = vec3(0., 1., 0.);
	// vec3 speciesCColour = vec3(0., 0., 1.);
	// vec3 speciesDColour = vec3(1., 1., 0.);

    vec3 light_falloff = vec3(1.);

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

    // colour.gb = trail.rg;

    out_color = vec4(colour, 1.);
}