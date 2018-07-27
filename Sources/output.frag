// dof based on: http://pasteall.org/10779

#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D postProcessOutput;
uniform sampler2D forwardDepthTex;

uniform float exposure;
uniform vec3 gamma;
uniform bool uncharted_tonemap;
uniform bool depth_of_field;
uniform int height;
uniform int width;
uniform float focus;

const float blurclamp = 3.0;
const float bias = 0.6; 

vec3 UnchartedTonemap(vec3 color);
vec3 GammaCorrect(vec3 color);
vec3 DepthOfField(float factor);

  const float A = 0.15f;
  const float B = 0.50f;
  const float C = 0.10f;
  const float D = 0.20f;
  const float E = 0.02f;
  const float F = 0.30f;
  const float W = 11.2f;
  const float E_over_F = E/F;

void main()
{   
	// input sample
	vec3 color;
	vec4 depth1 = texture2D(forwardDepthTex, TexCoords.st );
    float factor = ( depth1.x - focus );
	if(depth_of_field && (factor > 0.01 || factor < -0.01)){
		color = DepthOfField(factor);
	} else {
	 color = texture(postProcessOutput, TexCoords.st).rgb;
	}

	if(uncharted_tonemap){
		color = UnchartedTonemap(color);
		vec3 whiteScale = 1.0f / UnchartedTonemap(vec3(W));
		color *= whiteScale;
	}
            
	color = vec3(1.0) - exp(-color * exposure);
	FragColor = vec4(GammaCorrect(color), 1.0f);

}

// HDR and Gamma
vec3 GammaCorrect(vec3 color){
	return pow(color, gamma);
}

//Uncharted Tonemap
//Uncharted Tonemap based on: https://github.com/JoshuaSenouf/GLEngine/blob/master/resources/shaders/postprocess/firstpass.frag
vec3 UnchartedTonemap(vec3 color)
{
  return ((color * (A * color + C * B) + D * E) / (color * ( A * color + B) + D * F)) - E_over_F;
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