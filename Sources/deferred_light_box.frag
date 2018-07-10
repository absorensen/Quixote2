#version 330 core
layout (location = 1) out vec3 forwardContrib;

uniform vec3 lightColor;

out vec4 FragColor;


void main()
{
	forwardContrib = lightColor;
    FragColor = vec4(lightColor, 1.0);
}