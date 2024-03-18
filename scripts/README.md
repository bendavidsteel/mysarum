Sage Jensen Physarum shaders

```glsl
precision highp float; 

uniform sampler2D u_trail; 

in vec2 i_P; 
in float i_A; 
in float i_T; 

out vec2 v_P; 
out float v_A; 
out float v_T; 

uniform vec2 i_dim; 
uniform int pen; 
uniform float[19] v; 
uniform float[8] mps; 
uniform int frame; 

vec2 bd(vec2 pos) { 
    pos *= .5; 
    pos += vec2(.5); 
    pos -= floor(pos); 
    pos -= vec2(.5); 
    pos *= 2.; 
    return pos; 
} 

float gn(in vec2 coordinate, in float seed) { 
    return fract(tan(distance(coordinate*(seed+0.118446744073709551614), vec2(0.118446744073709551614, 0.314159265358979323846264)))*0.141421356237309504880169); 
} 

vec2 cr(float t) { 
    vec2 G1 = vec2(mps[0], mps[1]); 
    vec2 G2 = vec2(mps[2], mps[3]); 
    vec2 G3 = vec2(mps[4], mps[5]); 
    vec2 G4 = vec2(mps[6], mps[7]); 
    vec2 A = G1*-0.5+G2*1.5+G3*-1.5+G4*0.5; 
    vec2 B = G1+G2*-2.5+G3*2.+G4*-.5; 
    vec2 C = G1*-0.5+G3*0.5 ; 
    vec2 D = G2; 
    return t*(t*(t*A+B)+C)+D; 
} 

void main() { 
    vec2 dir = vec2(cos(i_T), sin(i_T)); 
    float hd= i_dim.x/2.; 
    vec2 sp=.5*(i_P+ vec2(1.0)); 
    // extra sensor location 
    float extra_sensor_value= texture(u_trail, bd(sp + (v[13]/hd)*dir + vec2(0.,v[12]/hd))).x; 
    extra_sensor_value= max(extra_sensor_value, 0.000000001); 
    // sensor distance can be increased depending on concentration in extra sensor location
    float sensor_distance=v[0]/hd + v[2]*pow(extra_sensor_value,v[1])*250./hd; 
    float speed=v[9]/hd + v[11]*pow(extra_sensor_value,v[10])*250./hd; 
    // sensor angle can be increased depending on concentration in extra sensor location
    float sensor_angle = v[3] + v[5]*pow(extra_sensor_value, v[4]); 
    float turn_angle=v[6] + v[8]*pow(extra_sensor_value, v[7]); 
    float m=texture(u_trail, bd(sp+ sensor_distance*vec2(cos(i_T), sin(i_T)))).x; 
    float l=texture(u_trail, bd(sp+ sensor_distance*vec2(cos(i_T+sensor_angle), sin(i_T+sensor_angle)))).x; 
    float r=texture(u_trail, bd(sp+ sensor_distance*vec2(cos(i_T-sensor_angle), sin(i_T-sensor_angle)))).x; 
    float h=i_T; 
    if (m>l&&m>r){} 
    else if (m<l&&m<r){
        // randomly turn if ahead has the lowest concentration
        if (gn(i_P*1332.4324,i_T) > 0.5) h+= turn_angle; 
        else h-=turn_angle;
    } 
    else if (l<r) h-=turn_angle; 
    else if (l>r) h+=turn_angle; 
    vec2 new_direction=vec2(cos(h), sin(h)); 
    vec2 op=i_P + new_direction * speed; 
    const float segmentPop=0.0005; 
    if (pen==1&&i_A<segmentPop){ 
        op=2.*cr(i_A/segmentPop)-vec2(1.); 
        op+= new_direction*pow(gn(i_P*132.43,i_T), 1.8); 
    } 
    v_P = bd(op); 
    v_A= fract(i_A+segmentPop); 
    v_T =h; 
}```