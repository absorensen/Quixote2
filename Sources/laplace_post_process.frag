#version 430 core
layout (location = 0) out vec3 reconstructionPostProcessOut;
//layout (location = 0) out vec3 reconstructionPostProcessOutRB;
//layout (location = 1) out vec3 reconstructionPostProcessOutG;

out vec4 FragColor;


in vec2 TexCoords;

uniform sampler2D deferredOutput;
uniform sampler2D forwardOutput;

uniform bool edges;
uniform bool transparency;

const float offset = 1.0 / 512.0;  

const vec2 offsets[5] = vec2[](
        vec2( 0.0f,    0.0f),   // center-center
        vec2( 0.0f,    offset), // top-center
        vec2( 0.0f,   -offset), // bottom-center
        vec2(-offset,  0.0f),   // center-left
        vec2( offset,  0.0f)   // center-right
    );


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

		col = 4*texSamples[0];
		for(int i = 1; i < 5; i++)
			col += -texSamples[i];

	} else {
		col = vec3(texture(forwardOutput, TexCoords.st));
		deferred = texture(deferredOutput, TexCoords.st).rgb;
		if(transparency) col = col.x > 0 && col.y > 0 && col.z > 0 ? col + col * deferred : deferred;
		else col = col.x > 0 && col.y > 0 && col.z > 0 ? col + col * deferred : deferred;
	}
	col.x = col.x > 0.0 ? col.x : 0.0;
	col.y = col.y > 0.0 ? col.y : 0.0;
	col.z = col.z > 0.0 ? col.z : 0.0;
	reconstructionPostProcessOut = col;
	FragColor = vec4(col, 1.0);
}