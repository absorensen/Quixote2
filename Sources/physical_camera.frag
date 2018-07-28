// dof based on: https://github.com/orthecreedence/ghostie/blob/master/opengl/glsl/dof.bokeh.2.4.frag
#version 430 core
layout (location = 0) out vec3 cameraOutput;

out vec4 FragColor;

smooth in vec2 TexCoords;

uniform sampler2D postProcessOutput;
uniform sampler2D forwardDepthTex;

#define PI  3.14159265

uniform int height;
uniform int width;
uniform float focalDepth;
uniform bool depth_of_field;
uniform float znear;
uniform float zfar;
uniform bool autofocus;
uniform bool showfocus;
uniform bool manualdof; //manual dof calculation

vec2 texel = vec2(1.0/width, 1.0/height);

float focalLength = 45.0;
float fstop = 1.4;

int samples = 3; //samples on the first ring
int rings = 3; //ring count

float ndofstart = 1.0; //near dof blur start
float ndofdist = 1.0; //near dof blur falloff distance
float fdofstart = 1.0; //far dof blur start
float fdofdist = 1.5; //far dof blur falloff distance

float CoC = 0.03;//circle of confusion size in mm (35mm film = 0.03mm)

bool vignetting = false; //use optical lens vignetting?
float vignout = 1.3; //vignetting outer border
float vignin = 0.0; //vignetting inner border
float vignfade = 22.0; //f-stops till vignete fades

//bool autofocus = true; //use autofocus in shader? disable if you use external focalDepth value
vec2 focus = vec2(0.5,0.5); // autofocus point on screen (0.0,0.0 - left lower corner, 1.0,1.0 - upper right)
float maxblur = 1.0; //clamp value of max blur (0.0 = no blur,1.0 default)

float threshold = 6.7; //highlight threshold;
float gain = 100.0; //highlight gain;

float bias = 0.5; //bokeh edge bias
float fringe = 0.7; //bokeh chromatic aberration/fringing

bool noise = false; //use noise instead of pattern for sample dithering
float namount = 0.0001; //dither amount

bool depthblur = true; //blur the depth buffer?
float dbsize = 1.25; //depthblursize



bool pentagon = true; //use pentagon as bokeh shape?
float feather = 0.4; //pentagon shape feather

vec3 phys_dof();
float penta_dof(vec2 coord);
vec3 color(vec2 coords,float blur);
float bdepth(vec2 coords);
vec2 rand(vec2 coord);
vec3 debugFocus(vec3 col, float blur, float depth);
float linearize(float depth);
float vignette();

void main()
{   
	vec3 color;
	if(depth_of_field){
		color = phys_dof();
	} else {
		color = texture(postProcessOutput, TexCoords.st).rgb;
	}

	cameraOutput = color;
}

vec3 phys_dof(){
	float depth = linearize(texture2D(forwardDepthTex, TexCoords.st).x);
	
	if (depthblur)
	{
		depth = linearize(bdepth(TexCoords.st));
	}
	
	//focal plane calculation
	
	float fDepth = focalDepth;
	
	if (autofocus)
	{
		fDepth = linearize(texture2D(forwardDepthTex,focus).x);
	}
	
	//dof blur factor calculation
	
	float blur = 0.0;
	
	if (manualdof)
	{    
		float a = depth-fDepth; //focal plane
		float b = (a-fdofstart)/fdofdist; //far DoF
		float c = (-a-ndofstart)/ndofdist; //near Dof
		blur = (a>0.0)?b:c;
	}
	
	else
	{
		float f = focalLength; //focal length in mm
		float d = fDepth*1000.0; //focal plane in mm
		float o = depth*1000.0; //depth in mm
		
		float a = (o*f)/(o-f); 
		float b = (d*f)/(d-f); 
		float c = (d-f)/(d*fstop*CoC); 
		
		blur = abs(a-b)*c;
	}
	
	blur = clamp(blur,0.0,1.0);
	
	// calculation of pattern for ditering
	
	vec2 noise = rand(TexCoords.st)*namount*blur;
	
	// getting blur x and y step factor
	
//	float w = (1.0/width)*blur*maxblur+noise.x;
//	float h = (1.0/height)*blur*maxblur+noise.y;
	float w = texel.x*blur*maxblur+noise.x;
	float h = texel.y*blur*maxblur+noise.y;
	
	// calculation of final color
	
	vec3 col = vec3(0.0);
	
	if(blur < 0.05) //some optimization thingy
	{
		col = texture2D(postProcessOutput, TexCoords.st).rgb;
	}
	
	else
	{
		col = texture2D(postProcessOutput, TexCoords.st).rgb;
		float s = 1.0;
		int ringsamples;
		float i_f = 1.0;
		float j_f = 0.0;
		float ringsamples_f;
		float one_m_ringsamples_f;
		float rings_f = float(rings);
		for (int i = 1; i <= rings; ++i)
		{   
			ringsamples = i * samples;
			ringsamples_f = i_f * samples;
			one_m_ringsamples_f = 1.0/ringsamples_f;
			for (int j = 0 ; j < ringsamples ; ++j)   
			{
				float step = PI*2.0 * one_m_ringsamples_f;
				float pw = (cos(j_f*step)*i_f);
				float ph = (sin(j_f*step)*i_f);
				float p = 1.0;
				if (pentagon)
				{ 
					p = penta_dof(vec2(pw,ph));
				}
				col += color(TexCoords.st + vec2(pw*w,ph*h),blur)*mix(1.0,(i_f)/(rings_f),bias)*p;  
				s += 1.0*mix(1.0,(i_f)/(rings_f),bias)*p;   
				j_f += 1.0;
			}
			j_f = 0.0;
			i_f += 1.0;
		}
		col /= s; //divide by sample count
	}
	
	if (showfocus)
	{
		col = debugFocus(col, blur, depth);
	}
	
	if (vignetting)
	{
		col *= vignette();
	}
	return col;
}

float vignette()
{
//TODO: Ask how to translate
//	float dist = distance(TexCoords[3].st, vec2(0.5,0.5));
	float dist = distance(TexCoords.st, vec2(0.5,0.5));
	float one_vignfade = 1.0/vignfade;
	dist = smoothstep(vignout+(fstop*one_vignfade), vignin+(fstop*one_vignfade), dist);
	return clamp(dist,0.0,1.0);
}

float linearize(float depth)
{
	return -zfar * znear / (depth * (zfar - znear) - zfar);
}

vec3 debugFocus(vec3 col, float blur, float depth)
{
	float edge = 0.002*depth; //distance based edge smoothing
	float m = clamp(smoothstep(0.0,edge,blur),0.0,1.0);
	float e = clamp(smoothstep(1.0-edge,1.0,blur),0.0,1.0);
	
	col = mix(col,vec3(1.0,0.5,0.0),(1.0-m)*0.6);
	col = mix(col,vec3(0.0,0.5,1.0),((1.0-e)-(1.0-m))*0.2);

	return col;
}

vec2 rand(vec2 coord) //generating noise/pattern texture for dithering
{
	float noiseX = ((fract(1.0-coord.s*(width*0.5))*0.25)+(fract(coord.t*(height*0.5))*0.75))*2.0-1.0;
	float noiseY = ((fract(1.0-coord.s*(width*0.5))*0.75)+(fract(coord.t*(height*0.5))*0.25))*2.0-1.0;
	
	if (noise)
	{
		noiseX = clamp(fract(sin(dot(coord ,vec2(12.9898,78.233))) * 43758.5453),0.0,1.0)*2.0-1.0;
		noiseY = clamp(fract(sin(dot(coord ,vec2(12.9898,78.233)*2.0)) * 43758.5453),0.0,1.0)*2.0-1.0;
	}
	return vec2(noiseX,noiseY);
}

float penta_dof(vec2 coords) //pentagonal shape
{
	float scale = float(rings) - 1.3;
	vec4  HS0 = vec4( 1.0,         0.0,         0.0,  1.0);
	vec4  HS1 = vec4( 0.309016994, 0.951056516, 0.0,  1.0);
	vec4  HS2 = vec4(-0.809016994, 0.587785252, 0.0,  1.0);
	vec4  HS3 = vec4(-0.809016994,-0.587785252, 0.0,  1.0);
	vec4  HS4 = vec4( 0.309016994,-0.951056516, 0.0,  1.0);
	vec4  HS5 = vec4( 0.0        ,0.0         , 1.0,  1.0);
	
	vec4  one = vec4( 1.0 );
	
	vec4 P = vec4((coords),vec2(scale, scale)); 
	
	vec4 dist = vec4(0.0);
	float inorout = -4.0;
	
	dist.x = dot( P, HS0 );
	dist.y = dot( P, HS1 );
	dist.z = dot( P, HS2 );
	dist.w = dot( P, HS3 );
	
	dist = smoothstep( -feather, feather, dist );
	
	inorout += dot( dist, one );
	
	dist.x = dot( P, HS4 );
	dist.y = HS5.w - abs( P.z );
	
	dist = smoothstep( -feather, feather, dist );
	inorout += dist.x;
	
	return clamp( inorout, 0.0, 1.0 );
}


float bdepth(vec2 coords) //blurring depth
{
	float d = 0.0;
	float kernel[9];
	vec2 offset[9];
	
	vec2 wh = vec2(texel.x, texel.y) * dbsize;
	
	offset[0] = vec2(-wh.x,-wh.y);
	offset[1] = vec2( 0.0, -wh.y);
	offset[2] = vec2( wh.x -wh.y);
	
	offset[3] = vec2(-wh.x,  0.0);
	offset[4] = vec2( 0.0,   0.0);
	offset[5] = vec2( wh.x,  0.0);
	
	offset[6] = vec2(-wh.x, wh.y);
	offset[7] = vec2( 0.0,  wh.y);
	offset[8] = vec2( wh.x, wh.y);
	
	kernel[0] = 1.0/16.0;   kernel[1] = 2.0/16.0;   kernel[2] = 1.0/16.0;
	kernel[3] = 2.0/16.0;   kernel[4] = 4.0/16.0;   kernel[5] = 2.0/16.0;
	kernel[6] = 1.0/16.0;   kernel[7] = 2.0/16.0;   kernel[8] = 1.0/16.0;
	
	
	for( int i=0; i<9; i++ )
	{
		float tmp = texture2D(forwardDepthTex, coords + offset[i]).r;
		d += tmp * kernel[i];
	}
	
	return d;
}

vec3 color(vec2 coords,float blur) //processing the sample
{
	vec3 col = vec3(0.0);
	
	col.r = texture2D(postProcessOutput,coords + vec2(0.0,1.0)*texel*fringe*blur).r;
	col.g = texture2D(postProcessOutput,coords + vec2(-0.866,-0.5)*texel*fringe*blur).g;
	col.b = texture2D(postProcessOutput,coords + vec2(0.866,-0.5)*texel*fringe*blur).b;
	
	vec3 lumcoeff = vec3(0.299,0.587,0.114);
	float lum = dot(col.rgb, lumcoeff);
	float thresh = max((lum-threshold)*gain, 0.0);
	return col+mix(vec3(0.0),col,thresh*blur);
}