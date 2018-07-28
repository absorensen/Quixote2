#version 430 core
layout (location = 0) out vec3 reconstructionPostProcessOut;

out vec4 FragColor;


in vec2 TexCoords;

uniform sampler2D deferredOutput;
uniform sampler2D forwardOutput;

uniform bool edges;
uniform bool transparency;
uniform int height;
uniform int width;

const vec2 offset = vec2(1.0 / width, 1.0/height);  

const vec2 offsets[5] = vec2[](
        vec2( 0.0f,    0.0f),   // center-center
        vec2( 0.0f,    offset.y), // top-center
        vec2( 0.0f,   -offset.y), // bottom-center
        vec2(-offset.x,  0.0f),   // center-left
        vec2( offset.x,  0.0f)   // center-right
    );

vec3 GammaCorrect(vec3 color);
void main()
{             
	vec3 col;
	vec3 deferred;
	if (edges){
		vec3 texSamples[5];
		for(int i = 0; i < 5; ++i){
			texSamples[i] = vec3(texture(forwardOutput, TexCoords.st + offsets[i]));
			deferred = texture(deferredOutput, TexCoords.st + offsets[i]).rgb;
			if(transparency) texSamples[i] = texSamples[i].x > 0 && texSamples[i].y > 0 && texSamples[i].z > 0 ? texSamples[i] + texSamples[i] * deferred : deferred;
			else texSamples[i] = texSamples[i].x > 0 && texSamples[i].y > 0 && texSamples[i].z > 0 ? texSamples[i] : deferred;
		}	

		col -= 4.0*texSamples[0];
		col = texSamples[1];
		col = texSamples[2];
		col = texSamples[3];
		col = texSamples[4];

	} else {
		col = vec3(texture(forwardOutput, TexCoords.st));
		deferred = texture(deferredOutput, TexCoords.st).rgb;
		if(transparency) col = col.x > 0 && col.y > 0 && col.z > 0 ? col + col * deferred : deferred;
		else col = col.x > 0 && col.y > 0 && col.z > 0 ? col + col * deferred : deferred;
	}

	reconstructionPostProcessOut = col;
	FragColor = vec4(col, 1.0);
}
