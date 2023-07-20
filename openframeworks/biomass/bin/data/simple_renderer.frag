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
layout(rg16,binding=4) uniform restrict image2D optFlowMap;
layout(rg16,binding=0) uniform restrict image2D reactionMap;
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
uniform int opticalFlowDownScale;
uniform int display;

out vec4 out_color;

void main()
{
	ivec2 coord = ivec2(gl_FragCoord.xy);
	vec2 pos = vec2(coord);

	float dist_to_light = distance(pos.xy, light.xy);
	vec2 dir_to_light = normalize(light.xy - pos.xy);

	vec2 audio = imageLoad(audioMap, coord).xy;
	vec3 audioColour = colourC * audio.x;
	audioColour += colourD * audio.y;
	vec3 lighting = vec3(1.);
	float falloff = resolution.x / 3;
	vec3 light_falloff = vec3(exp(-dist_to_light / (50 * falloff))) + 0.8 * audioColour;

	vec3 colour = vec3(0.);

	if (display == 4) {
		vec2 opticalFlowForce = imageLoad(optFlowMap, coord / opticalFlowDownScale).xy;
		opticalFlowForce = (2 * opticalFlowForce) - 1; // convert to -1-1 range
		colour = vec3(opticalFlowForce.x, opticalFlowForce.y, -opticalFlowForce.x);

	} else if (display == 3) {
		colour = audioColour;

	} else if (display == 2) {
		vec2 chems = imageLoad(reactionMap, coord).xy;

		float this_chem_height = chem_height * (1 + length(audio));

		float pos_height = chems.y * this_chem_height;
		
		float height_to_light = light.z - pos_height;
		float max_dist_to_other_peak = dist_to_light * (this_chem_height - pos_height) / height_to_light;
		
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

		// colour = chems.x * colourA * lighting;
		// colour += chems.y * colourB * lighting;

	} else if (display == 1) {
		float pos_height = 0.;
		float height_to_light = light.z - pos_height;
        float trail_d = dist_to_light * (trail_height - pos_height) / height_to_light;
        vec2 trail_coord = pos.xy + (dir_to_light * trail_d);
        vec4 trail_shadow = imageLoad(trailMap, ivec2(trail_coord));

		vec3 flow = texture(flowMap, coord).xyz;
		colour = 0.2 * flow;

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
    }

    out_color = vec4(colour, 1.);
}