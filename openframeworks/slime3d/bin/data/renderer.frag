#version 440

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(std140, binding=1) buffer species{
    Species allSpecies[];
};

layout(rgba8,binding=4) uniform restrict readonly image3D trailMap;

layout(rgba8,binding=5) uniform restrict image3D flowMap;

uniform ivec2 screen_res;
uniform ivec3 trail_res;
uniform float boxScale;

out vec4 out_color;

vec3 getTexCoord(vec3 pos)
{
	vec3 uv = pos + vec3(0.5, 0.5, 0.5);
	// uv *= boxScale;
	uv *= trail_res.xyz;
	return uv;
}

void main()
{
	vec2 coord = gl_FragCoord.xy / screen_res.xy;

	coord = coord * trail_res.xz;

	for (int i = 0; i <= trail_res.y; i++) {
		out_color += imageLoad(trailMap, ivec3(coord.x, i, coord.y)).r * vec4(1., 1., 1., 1.);
	}

	// vec2 centre = screen_res.xy * 0.5 / screen_res.y;
	// vec2 from_centre = coord - centre;

	// vec3 rayOrigin = vec3(from_centre, 0.);
	// vec3 rayDir = vec3(0., 0., 1.);
	// // vec3 raySink = vec3(0., 0., 10.);
	// // vec3 rayDir = raySink - rayOrigin;
	// rayDir = normalize(rayDir);

	// int maxDist = trail_res.z;

	// vec4 col_acc = vec4(0., 0., 0., 0.);
	// vec4 trail_colour = vec4(0., 0., 0., 0.);
	// for (int i = 0; i < maxDist; i++)
	// {
	// 	vec3 texCoord = getTexCoord(rayOrigin + rayDir * float(i));
	// 	vec4 trail_sample = imageLoad(trailMap, ivec3(texCoord));

	// 	trail_colour = trail_sample.r * allSpecies[0].colour;
	// 	trail_colour += trail_sample.g * allSpecies[1].colour;
	// 	trail_colour += trail_sample.b * allSpecies[2].colour;
	// 	trail_colour += trail_sample.a * allSpecies[3].colour;
	// 	trail_colour = min(trail_colour, vec4(1., 1., 1., 1.));
	// 	col_acc = trail_colour;
	// 	// float oneMinusAlpha = 1. - col_acc.a;
	// 	// // trail_colour *= 0.1;
	// 	// col_acc.rgb = mix(col_acc.rgb, trail_colour.rgb * trail_colour.a, oneMinusAlpha);
	// 	// col_acc.a += trail_colour.a * oneMinusAlpha;
	// 	// col_acc.rgb /= col_acc.a;
	// 	// if (col_acc.a >= 1.0)
	// 	// {
	// 	// 	break; // terminate if opacity > 1
	// 	// }
	// }
	// out_color = col_acc;
}