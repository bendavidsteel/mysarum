OF_GLSL_SHADER_HEADER

// these are for the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4 position;

layout(rg16,binding=0) uniform restrict image2D reactionMap;

// the time value is passed into the shader by the OF app.
uniform float time;

void main()
{
    vec2 chems = imageLoad(reactionMap, coord).xy;
	
    vec4 modifiedPosition = modelViewProjectionMatrix * position;
	modifiedPosition.y += displacementY;
	gl_Position = modifiedPosition;
}