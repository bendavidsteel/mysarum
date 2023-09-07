#version 440

uniform sampler2DRect reactionMap;
uniform sampler2DRect audioMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;

out vec4 out_color;

void main()
{
	vec2 coord = gl_FragCoord.xy;
	vec2 pos = vec2(coord);

	float dist_to_light = distance(pos.xy, light.xy);
	vec2 dir_to_light = normalize(light.xy - pos.xy);

	vec2 audio = texture(audioMap, coord).xy;
	vec3 audioColour = colourC * audio.x;
	audioColour += colourD * audio.y;
	vec3 lighting = vec3(1.);
	float falloff = resolution.x / 3;
	vec3 light_falloff = vec3(exp(-dist_to_light / (50 * falloff))) + 0.8 * audioColour;

	vec3 colour = vec3(0.);

    vec2 chems = texture(reactionMap, coord).xy;

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

        float other_peak_height = texture(reactionMap, ivec2(other_peak)).y * this_chem_height;
        float light_height = pos_height + (dist * height_to_light / dist_to_light);
        if (other_peak_height > light_height) {
            // in shadow
            lighting *= vec3(0.4);
            break;
        }
    }
    lighting *= light_falloff;

    colour = chems.x * colourA * lighting;
    colour += chems.y * colourB * lighting;

    out_color = vec4(colour, 1.);
}