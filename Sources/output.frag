#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D postProcessOutput;

uniform float exposure;
uniform vec3 gamma;
void main()
{   
	// input sample
	vec3 inputSample = texture(postProcessOutput, TexCoords.st).rgb;

	// HDR and Gamma
	vec3 result = vec3(1.0) - exp(-inputSample * exposure);
    result = pow(result, gamma);
    FragColor = vec4(result, 1.0);
}