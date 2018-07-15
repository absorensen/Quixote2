#extension GL_ARB_texture_rectangle : enable
uniform sampler2DRect fft;
uniform sampler2DRect scrambler;
uniform sampler2DRect real_weight;
uniform sampler2DRect imag_weight;
uniform int dimension;
uniform bvec2 inverse;

void main()
{
	vec2 texcoord = gl_FragCoord.xy - vec2(0.5, 0.5);
	vec4 i = texture2DRect(scrambler, texcoord);
	vec4 wr = texture2DRect(real_weight, texcoord);
	vec4 wi = texture2DRect(imag_weight, texcoord);
	vec2 newcoord;
	newcoord.x = dimension == 0 ? i.r : texcoord.x;
	newcoord.y = dimension == 1 ? i.r : texcoord.y;
	vec4 Input1 = texture2DRect(fft, newcoord);
	newcoord.x = dimension == 0 ? i.a : texcoord.x;
	newcoord.y = dimension == 1 ? i.a : texcoord.y;
	vec4 Input2 = texture2DRect(fft, newcoord);
	vec4 Res;
	float imag = inverse.x ? -wi.a : wi.a;
	Res.x = wr.a*Input2.x - imag*Input2.y;
	Res.y = imag*Input2.x + wr.a*Input2.y;
	imag = inverse.y ? -wi.a : wi.a;
	Res.z = wr.a*Input2.z - imag*Input2.w;
	Res.w = imag*Input2.z + wr.a*Input2.w;
	Res += Input1;
	Res.xy = inverse.x ? Res.xy/2.0 : Res.xy;
	Res.zw = inverse.y ? Res.zw/2.0 : Res.zw;
	gl_FragColor = Res;
}