#extension GL_ARB_texture_rectangle : enable
varying vec2 position;
uniform sampler2DRect fft;

void main()
{
	vec4 ffts = texture2DRect(fft, position.xy);
    gl_FragColor.rgb = vec3(length(ffts.rg));
}