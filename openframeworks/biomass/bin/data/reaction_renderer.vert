#version 440

uniform sampler2DRect reactionMap;
uniform sampler2DRect audioMap;

uniform vec3 colourA;
uniform vec3 colourB;
uniform vec3 colourC;
uniform vec3 colourD;
uniform ivec2 resolution;
uniform vec3 light;
uniform float chemHeight;

// these are from the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4 position;
in vec2 texcoord;

out vec2 texCoordVarying;

float easeOut(float x) {
    float factor = 10.;
    return 1.0 - pow(1.0 - x, factor);
}

void main()
{
    // get the position of the vertex relative to the modelViewProjectionMatrix
    vec4 modifiedPosition = modelViewProjectionMatrix * position;

    // here we get the red channel value from the texture
    // to use it as vertical displacement
    float displacementY = texture(reactionMap, texcoord).y;

    // use the displacement we created from the texture data
    // to modify the vertex position
	modifiedPosition.y -= easeOut(displacementY) * chemHeight;
	
    // this is the resulting vertex position
    gl_Position = modifiedPosition;

    // pass the texture coordinates to the fragment shader
    texCoordVarying = texcoord;
}
