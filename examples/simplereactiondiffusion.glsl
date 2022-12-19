void mainImage( out vec4 fragColour, in vec2 fragCoord )
{
    ivec2 coord = ivec2(fragCoord);
    ivec2 res = ivec2(iResolution.xy);
    
    #define Wrap(c) ((c+res)%res)
    
    fragColour = texelFetch(iChannel0,coord,0);
    
	// reaction diffusion
    // all of these params can vary with position/time/whatever to get nice effects
    float laplaceSelf = -1.;
    float laplaceAdjacent = .2;
    float laplaceDiagonal = .05;
    float deltaT = 1.; // could use frame duration but I'm pretty sure the maths isn't really linear
    vec2 diffusionRate = vec2(1,.5);
    
    vec2 uv = fragCoord/iResolution.xy;

    // fead & kill rates
    vec2 feedKill = mix( vec2(0,.04), vec2(.1,.07), vec2(.23,.5) ); // terminating worms
// some pretty looking alternatives:
//    vec2 feedKill = vec2(.033,.063);// dividing bacteria (needs sharp shapes)
//    vec2 feedKill = vec2(.023,.053) + uv.yx*vec2(0,.01); // spots! (variations by adjusting kill)
//    vec2 feedKill = vec2(.031,.058); // fungal
//    vec2 feedKill = vec2(.025,.055); // angle sprouts
//    vec2 feedKill = mix( vec2(.02,.04), vec2(.0,.05), uv ); // weirdsmoke
//    vec2 feedKill = mix( vec2(.03,.03), vec2(.0,.06), uv.yx ); // weirdsmoke
//    vec2 feedKill = vec2(uv.x*uv.y,uv.y)*vec2(.02,.06); // smoke waves
//    vec2 feedKill = vec2(.4*.6,.6)*vec2(.02,.06); // smooth arcs
//    vec2 feedKill = vec2(.8*.6,.6)*vec2(.02,.06); // more spirally
//    vec2 feedKill = vec2(.6,.65)*vec2(.02,.06); // cycling spirals
//    vec2 feedKill = vec2(.75,.8)*vec2(.02,.06); // spiral puffs
//    vec2 feedKill = vec2(.02,.055); // plankton
//    vec2 feedKill = mix( vec2(0,.04), vec2(.1,.07), vec2(.21,.37) );// constant growth, different features
//    vec2 feedKill = mix( vec2(0,.04), vec2(.1,.07), vec2(.2,.33) ); // very active
//    vec2 feedKill = mix( vec2(0,.04), vec2(.1,.07), uv ); // map
    
    vec2 AB = fragColour.xy;

    // sample neighbouring pixels and apply weights to them
    ivec3 d = ivec3(-1,0,1);
    vec2 laplace = laplaceSelf * AB;
    laplace += 
        (
            texelFetch(iChannel0,Wrap(coord+d.xy),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.zy),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.yx),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.yz),0).xy
        )*laplaceAdjacent;
    laplace += 
        (
            texelFetch(iChannel0,Wrap(coord+d.xx),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.xz),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.zx),0).xy +
            texelFetch(iChannel0,Wrap(coord+d.zz),0).xy
        )*laplaceDiagonal;
    
    vec2 deltaAB = diffusionRate*laplace;
    deltaAB += vec2(-1,1)*AB.x*AB.y*AB.y;
    deltaAB.x += feedKill.x*(1.-AB.x);
    deltaAB.y -= (feedKill.y+feedKill.x)*AB.y;
    
    AB += deltaT * deltaAB;
    
    AB = clamp(AB,0.,1.);
    
    fragColour.xy = AB;
    
    if ( iFrame == 0 )
    {
    	fragColour = vec4(1,0,0,1);
        fragColour.y = smoothstep(4.,3.,min(min(length(fragCoord-iResolution.xy*.5),length(fragCoord-iResolution.xy*vec2(.6,.3))),length(fragCoord-iResolution.xy*.45) ));
        fragColour.x = 1.-fragColour.y;
    }
    
    // let mouse add more
    if ( iMouse.z > 0. )
    {
        fragColour.xy = mix( fragColour.xy, vec2(0,1), smoothstep(4.,3.,length(iMouse.xy-fragCoord.xy)) );
    }
}