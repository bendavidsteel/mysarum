#version 440

uniform mat4 modelViewProjectionMatrix;

in vec4 position;
// in vec2 texCoord;

// out vec2 texCoordVarying;

void main()
{
	gl_Position = modelViewProjectionMatrix * position;

	// texCoordVarying = texCoord;
}