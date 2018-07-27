#version 430 core
layout (location = 0) out vec3 forwardOutput;

uniform vec3 lightColor;

//out vec4 FragColor;

void main()
{
	forwardOutput = lightColor;
    //FragColor = vec4(lightColor, 1.0);
}