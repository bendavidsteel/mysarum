#version 440

uniform sampler2DRect audioMap;

// these are from the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4 position;
in vec2 texcoord;

out vec2 texCoordVarying;

void main()
{
    // get the position of the vertex relative to the modelViewProjectionMatrix
    vec4 modifiedPosition = modelViewProjectionMatrix * position;

    float audioHeight = 20.;

    // here we get the red channel value from the texture
    // to use it as vertical displacement
    float displacementY = texture(audioMap, texcoord).y;

    // use the displacement we created from the texture data
    // to modify the vertex position
	modifiedPosition.y += displacementY * audioHeight;
	
    // this is the resulting vertex position
    gl_Position = modifiedPosition;

    // pass the texture coordinates to the fragment shader
    texCoordVarying = texcoord;
}
