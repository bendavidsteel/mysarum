OF_GLSL_SHADER_HEADER

uniform float windStrength;
uniform vec2 windDirection;

// these are for the programmable pipeline system
uniform mat4 modelViewProjectionMatrix;
in vec4  position;
in vec4  color;
in float width;

out VS_OUT {
    vec4 color;
    float width;
} vs_out;

void main()
{
    vec4 modifiedPosition = modelViewProjectionMatrix * position;

    // wind
    modifiedPosition.x += windDirection.x * windStrength * position.y / width;
    modifiedPosition.y += windDirection.y * windStrength * position.y / width;

	gl_Position = modifiedPosition;
    vs_out.color = color;
    vs_out.width = width;
}
