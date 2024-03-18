OF_GLSL_SHADER_HEADER

layout(lines) in;
layout(triangle_strip, max_vertices = 20) out;

in VS_OUT {
    vec4 color;
    float width;
} gs_in[];  

uniform int numSegments = 10; // Set the number of segments

vec3 findOrthogonalVector(vec3 v) {
    // Find a vector that is orthogonal to v
    if (abs(v.y) < 0.9)
        return normalize(cross(v, vec3(0.0, 1.0, 0.0)));
    else
        return normalize(cross(v, vec3(0.0, 0.0, 1.0)));
}

void generateCylinderVertices(vec4 p0, vec4 p1) {
    vec3 cylinderAxis = p1.xyz - p0.xyz;
    float cylinderLength = length(cylinderAxis);
    vec3 cylinderDir = normalize(cylinderAxis);
    vec3 sideAVector = findOrthogonalVector(cylinderDir);
    vec3 sideBVector = cross(cylinderDir, sideAVector);
    
    for (int i = 0; i <= numSegments; i++) {
        float theta = 2.0 * 3.14159265358979323846 * float(i) / float(numSegments);
        vec3 unitPoint = (cos(theta) * sideAVector + sin(theta) * sideBVector);
        
        // Emit vertices
        vec4 vertex0 = p0 + vec4(gs_in[0].width * unitPoint, 0.0);
        vec4 vertex1 = p1 + vec4(gs_in[1].width * unitPoint, 0.0);
        
        // Emit two vertices for the triangle strip
        gl_Position = vertex0;
        EmitVertex();
        gl_Position = vertex1;
        EmitVertex();
    }
}

void main() {
    generateCylinderVertices(gl_in[0].gl_Position, gl_in[1].gl_Position);
    EndPrimitive();
}