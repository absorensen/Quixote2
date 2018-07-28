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

const float blurclamp = 3.0;
const float bias = 0.6; 

vec3 DepthOfField(float factor);

void main()
{   
	vec3 color;
	float depth1 = texture2D(forwardDepthTex, TexCoords.st ).r;
    float factor = ( depth1 - focus );
	if(depth_of_field && (factor > 0.01 || factor < -0.01)){
		color = DepthOfField(factor);
	} else {
		color = texture(postProcessOutput, TexCoords.st).rgb;
	}

	cameraOutput = color;
}

vec3 DepthOfField(float factor){

		float aspectratio = width/height;
        vec2 aspectcorrect = vec2(1.0,aspectratio);
        vec2 dofblur = vec2 (clamp( factor * bias, -blurclamp, blurclamp ));
        vec4 col = vec4(0.0);
       
        col += texture2D(postProcessOutput, TexCoords.st);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,0.4 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.15,0.37 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,0.29 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.37,0.15 )*aspectcorrect) * dofblur);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.4,0.0 )*aspectcorrect) * dofblur);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.37,-0.15 )*aspectcorrect) * dofblur);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,-0.29 )*aspectcorrect) * dofblur);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.15,-0.37 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,-0.4 )*aspectcorrect) * dofblur); 
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.15,0.37 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,0.29 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.37,0.15 )*aspectcorrect) * dofblur); 
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.4,0.0 )*aspectcorrect) * dofblur); 
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.37,-0.15 )*aspectcorrect) * dofblur);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,-0.29 )*aspectcorrect) * dofblur);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.15,-0.37 )*aspectcorrect) * dofblur);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.15,0.37 )*aspectcorrect) * dofblur*0.9);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.37,0.15 )*aspectcorrect) * dofblur*0.9);           
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.37,-0.15 )*aspectcorrect) * dofblur*0.9);           
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.15,-0.37 )*aspectcorrect) * dofblur*0.9);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.15,0.37 )*aspectcorrect) * dofblur*0.9);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.37,0.15 )*aspectcorrect) * dofblur*0.9);            
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.37,-0.15 )*aspectcorrect) * dofblur*0.9);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.15,-0.37 )*aspectcorrect) * dofblur*0.9);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,0.29 )*aspectcorrect) * dofblur*0.7);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.4,0.0 )*aspectcorrect) * dofblur*0.7);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,-0.29 )*aspectcorrect) * dofblur*0.7);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,-0.4 )*aspectcorrect) * dofblur*0.7);     
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,0.29 )*aspectcorrect) * dofblur*0.7);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.4,0.0 )*aspectcorrect) * dofblur*0.7);     
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,-0.29 )*aspectcorrect) * dofblur*0.7);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,0.4 )*aspectcorrect) * dofblur*0.7);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,0.29 )*aspectcorrect) * dofblur*0.4);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.4,0.0 )*aspectcorrect) * dofblur*0.4);       
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.29,-0.29 )*aspectcorrect) * dofblur*0.4);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,-0.4 )*aspectcorrect) * dofblur*0.4);     
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,0.29 )*aspectcorrect) * dofblur*0.4);
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.4,0.0 )*aspectcorrect) * dofblur*0.4);     
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( -0.29,-0.29 )*aspectcorrect) * dofblur*0.4);   
        col += texture2D(postProcessOutput, TexCoords.st + (vec2( 0.0,0.4 )*aspectcorrect) * dofblur*0.4);       
                       
        return col.xyz/41.0;
}