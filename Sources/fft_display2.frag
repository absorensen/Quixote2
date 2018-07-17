// based on an implementation by Jeppe Revall Frisvad
// url: http://www2.compute.dtu.dk/pubdb/views/publication_details.php?id=5771

#extension GL_ARB_texture_rectangle : enable
varying vec2 position;
uniform sampler2DRect fft;
uniform vec2 size;
uniform vec3 color;
void main()
{
	vec2 texcoord = mod(position, size);
    vec4 ffts = texture2DRect(fft, texcoord);
    gl_FragColor.r = length(ffts.ba);
    gl_FragColor.g = gl_FragColor.r;
    gl_FragColor.b = gl_FragColor.r;
    gl_FragColor.rgb *= color;
}