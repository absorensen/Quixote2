//based on https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/5.advanced_lighting/8.2.deferred_shading_volumes/8.2.deferred_light_box.fs

#version 430 core
layout (location = 0) out vec3 forwardOutput;

uniform vec3 lightColor;

void main()
{
	forwardOutput = lightColor;
}