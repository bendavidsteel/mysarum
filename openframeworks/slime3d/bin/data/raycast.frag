#version 440
//#extension GL_ARB_texture_rectangle : enable

in vec3 v_texcoord;
in vec3 v_cameraPosition;

out vec4 out_color;

struct Species{
	vec4 colour;
	vec4 sensorAttributes;
	vec4 movementAttributes;
};

layout(rgba8,binding=1) uniform restrict image3D trailMap;

layout(std140, binding=2) buffer species{
    Species allSpecies[];
};

uniform vec3 vol_d;
uniform vec3 vol_d_pot;
uniform vec2 bg_d;
uniform float zoffset;
uniform float quality;
uniform float threshold;
uniform float density;

struct Ray
{
    vec3 Origin;
    vec3 Dir;
};

struct BoundingBox
{
    vec3 Min;
    vec3 Max;
};

bool IntersectBox(Ray r, BoundingBox box, out float t0, out float t1)
{
    vec3 invR = 1.0 / r.Dir;
    vec3 tbot = invR * (box.Min - r.Origin);
    vec3 ttop = invR * (box.Max - r.Origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

void main()
{
    vec3 minv = vec3(0.) + 1. / vol_d_pot;
    vec3 maxv = (vol_d / vol_d_pot) - 1. / vol_d_pot;
    vec3 vec;
    vec3 vold = (maxv - minv) * vol_d;
    float vol_l = length(vold);

    vec4 col_acc = vec4(0, 0, 0, 0);
    vec3 zOffsetVec = vec3(0.0, 0.0, zoffset / vol_d_pot.z);
    vec3 backPos = v_texcoord;
    vec3 lookVec = normalize(backPos - v_cameraPosition);

    Ray eye = Ray(v_cameraPosition, lookVec);
    BoundingBox box = BoundingBox(vec3(0.), vec3(1.));

    float tnear, tfar;
    IntersectBox(eye, box, tnear, tfar);
    if (tnear < 0.15) tnear = 0.15;
    if (tnear > tfar) discard;

    vec3 rayStart = (eye.Origin + eye.Dir * tnear) * (maxv - minv) + minv; //vol_d/vol_d_pot;
    vec3 rayStop = (eye.Origin + eye.Dir * tfar) * (maxv - minv) + minv; //vol_d/vol_d_pot;

    vec3 dir = rayStop - rayStart; // starting position of the ray

    vec = rayStart;
    float dl = length(dir);
    if (dl == clamp(dl, 0., vol_l))
    {
        int steps = int(floor(length(vold * dir) * quality));
        vec3 delta_dir = dir / float(steps);
        vec4 trail_sample;
        vec4 trail_colour;
        float aScale = density / quality;

        float random = fract(sin(gl_FragCoord.x * 12.9898 + gl_FragCoord.y * 78.233) * 43758.5453);
        vec += delta_dir * random;

        //raycast
        for (int i = 0; i < steps; i++)
        {
            vec3 vecz = vec + zOffsetVec;
            if (vecz.z > maxv.z)
            {
                vecz.z -= maxv.z;
            }

            trail_sample = imageLoad(trailMap, ivec3(vecz * vol_d));
            if ((trail_sample.r + trail_sample.g + trail_sample.b + trail_sample.a) > threshold)
            {
                trail_colour = trail_sample.r * allSpecies[0].colour;
                trail_colour += trail_sample.g * allSpecies[1].colour;
                trail_colour += trail_sample.b * allSpecies[2].colour;
                trail_colour += trail_sample.a * allSpecies[3].colour;
                trail_colour = min(trail_colour, vec4(1., 1., 1., 1.));
                float oneMinusAlpha = 1. - col_acc.a;
                trail_colour *= aScale;
                col_acc.rgb = mix(col_acc.rgb, trail_colour.rgb * trail_colour.a, oneMinusAlpha);
                col_acc.a += trail_colour.a * oneMinusAlpha;
                col_acc.rgb /= col_acc.a;
                if (col_acc.a >= 1.0)
                {
                    break; // terminate if opacity > 1
                }
            }
            vec += delta_dir;
        }
    }

    out_color = col_acc;
}