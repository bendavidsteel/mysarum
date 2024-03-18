#version 440

uniform sampler2DRect reactionMap;

// these are from the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4 position;
in vec2 texcoord;

out vec2 texCoordVarying;

float getPosHeight(vec2 pos)
{
	float modHeight = 0.;
	return modHeight;
}

void main()
{
    // get the position of the vertex relative to the modelViewProjectionMatrix
    vec4 modifiedPosition = modelViewProjectionMatrix * position;

    float modHeight = getPosHeight(texcoord);

    // use the displacement we created from the texture data
    // to modify the vertex position
	modifiedPosition.y -= modHeight;
	
    // this is the resulting vertex position
    gl_Position = modifiedPosition;

    // pass the texture coordinates to the fragment shader
    texCoordVarying = texcoord;
}
