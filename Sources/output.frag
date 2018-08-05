// based on https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/5.advanced_lighting/6.hdr/6.hdr.fs
// and Uncharted Tonemap based on: https://github.com/JoshuaSenouf/GLEngine/blob/master/resources/shaders/postprocess/firstpass.frag

#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D inputTexture;
uniform sampler2D inputTexture2;

uniform float exposure;
uniform vec3 gamma;
uniform bool uncharted_tonemap;
uniform bool ssaoViz;
uniform bool difference;

vec3 UnchartedTonemap(vec3 color);
vec3 GammaCorrect(vec3 color);


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
	vec3 color = texture(inputTexture, TexCoords.st).rgb;

	if(uncharted_tonemap){
		color = UnchartedTonemap(color);
		vec3 whiteScale = 1.0f / UnchartedTonemap(vec3(W));
		color *= whiteScale;
	}
            
	color = vec3(1.0) - exp(-color * exposure);
	FragColor = vec4(GammaCorrect(color), 1.0f);

if(difference){
 color = texture(inputTexture2, TexCoords.st).rgb - texture(inputTexture, TexCoords.st).rgb;
 FragColor = vec4(color, 1.0f);
 }
}

// HDR and Gamma
vec3 GammaCorrect(vec3 color){
	return pow(color, gamma);
}

vec3 UnchartedTonemap(vec3 color)
{
  return ((color * (A * color + C * B) + D * E) / (color * ( A * color + B) + D * F)) - E_over_F;
}

