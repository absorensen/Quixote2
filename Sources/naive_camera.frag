// dof based on: http://pasteall.org/10779
#version 430 core
layout (location = 0) out vec3 cameraOutput;

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D postProcessOutput;
uniform sampler2D forwardDepthTex;

uniform int height;
uniform int width;
uniform float focus;
uniform bool depth_of_field;
uniform float znear;
uniform float zfar;

const float blurclamp = 3.0;
const float bias = 0.6; 
const float max_sample_depth = 60.0;
const float aspectratio = width/height;
const vec2 aspectcorrect = vec2(1.0,aspectratio);

vec3 DepthOfField(float depth, float factor);
float Linearize(float depth);
vec4 cal_sample(vec2 coord, vec2 dofblur, float depth);

void main()
{   
	vec3 color;
	float depth1 = texture2D(forwardDepthTex, TexCoords.st ).r;
    float factor = ( depth1 - focus );
	if(depth_of_field && (factor > 0.01 || factor < -0.01)){
		color = DepthOfField(depth1, factor);
	} else {
		color = texture(postProcessOutput, TexCoords.st).rgb;
	}

	cameraOutput = color;
}

float linearize(float depth)
{
	return -zfar * znear / (depth * (zfar - znear) - zfar);
}

vec4 cal_sample(vec2 coord, vec2 dofblur, float depth){
	coord = (vec2( coord )*aspectcorrect) * dofblur;
    vec4 samp = texture2D(postProcessOutput, TexCoords.st + coord);
	if (abs( linearize(texture2D(forwardDepthTex, TexCoords.st + coord ).r) - depth) < max_sample_depth) return samp;
	else return texture2D(postProcessOutput, TexCoords.st);

}

vec3 DepthOfField(float depth, float factor){
        vec2 blur = vec2 (clamp( factor * bias, -blurclamp, blurclamp ));
        vec4 col = vec4(0.0);
		depth = linearize(depth);

		col += texture2D(postProcessOutput, TexCoords.st);

		vec2 dofblur = blur;
		col += cal_sample(vec2( 0.0,0.4 ), dofblur, depth);
		col += cal_sample(vec2( 0.15,0.37 ), dofblur, depth);
		col += cal_sample(vec2( 0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.37,0.15 ), dofblur, depth);
		col += cal_sample(vec2( 0.4,0.0 ), dofblur, depth);
		col += cal_sample(vec2( 0.37,-0.15 ), dofblur, depth);
		col += cal_sample(vec2( 0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.15,-0.37 ), dofblur, depth);
		col += cal_sample(vec2( 0.0,-0.4 ), dofblur, depth);
		col += cal_sample(vec2( -0.15,0.37 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.37,0.15 ), dofblur, depth);
		col += cal_sample(vec2( -0.4,0.0 ), dofblur, depth);
        col += cal_sample(vec2( -0.37,-0.15 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.15,0.37 ), dofblur, depth);
		col += cal_sample(vec2( 0.15,-0.37 ), dofblur, depth);

		dofblur = blur * 0.9;
		col += cal_sample(vec2( 0.15,0.37 ), dofblur, depth);
		col += cal_sample(vec2( 0.15,-0.37 ), dofblur, depth);
		col += cal_sample(vec2( -0.15,-0.37 ), dofblur, depth);
		col += cal_sample(vec2( -0.15,0.37 ), dofblur, depth);
		col += cal_sample(vec2( 0.15,-0.37 ), dofblur, depth);
		col += cal_sample(vec2( -0.37,0.15 ), dofblur, depth);
		col += cal_sample(vec2( 0.37,0.15 ), dofblur, depth);
		col += cal_sample(vec2( 0.37,-0.15 ), dofblur, depth);
		col += cal_sample(vec2( -0.37,-0.15 ), dofblur, depth);
		
		dofblur = blur * 0.7;
		col += cal_sample(vec2( 0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.4,0.0 ), dofblur, depth);
		col += cal_sample(vec2( -0.4,0.0 ), dofblur, depth);
		col += cal_sample(vec2( 0.0,-0.4 ), dofblur, depth);
		col += cal_sample(vec2( 0.0,0.4 ), dofblur, depth);

		dofblur = blur * 0.4;
		col += cal_sample(vec2( 0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( -0.29,-0.29 ), dofblur, depth);
		col += cal_sample(vec2( 0.4,0.0 ), dofblur, depth);
		col += cal_sample(vec2( -0.4,0.0 ), dofblur, depth);
		col += cal_sample(vec2( 0.0,-0.4 ), dofblur, depth);
		col += cal_sample(vec2( 0.0,0.4 ), dofblur, depth);

        return col.xyz/41.0;
}