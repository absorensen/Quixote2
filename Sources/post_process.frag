#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D deferredContrib;
uniform sampler2D forwardContrib;

uniform float exposure;

void main()
{             
    const float gamma = 2.2;
    vec3 hdrColor = texture(forwardContrib, TexCoords).rgb;
	vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
    result = pow(result, vec3(1.0 / gamma));
    FragColor = vec4(hdrColor, 1.0);
}