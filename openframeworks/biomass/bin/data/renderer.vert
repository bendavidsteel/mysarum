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
layout(rg16,binding=7) uniform restrict image2D audioMap;

uniform sampler2DRect flowMap;
uniform sampler2DRect reactionMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chem_height;
uniform float trail_height;
uniform float time;
uniform float bps;

// these are from the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4 position;
in vec2 texcoord;

out vec2 texCoordVarying;

void main()
{
    // get the position of the vertex relative to the modelViewProjectionMatrix
    vec4 modifiedPosition = modelViewProjectionMatrix * position;
    
    // we need to scale up the values we get from the texture
    float scale = 100.;
    
    // here we get the red channel value from the texture
    // to use it as vertical displacement
    float displacementY = texture(tex0, texcoord).r;

    // use the displacement we created from the texture data
    // to modify the vertex position
	modifiedPosition.y += displacementY * scale;
	
    // this is the resulting vertex position
    gl_Position = modifiedPosition;

    // pass the texture coordinates to the fragment shader
    texCoordVarying = texcoord;
}
