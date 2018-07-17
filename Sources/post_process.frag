#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D deferredContrib;
uniform sampler2D forwardContrib;

uniform float exposure;
const float offset = 1.0 / 300.0;  

const vec2 offsets[5] = vec2[](
        vec2( 0.0f,    0.0f),   // center-center
        vec2( 0.0f,    offset), // top-center
        vec2( 0.0f,   -offset), // bottom-center
        vec2(-offset,  0.0f),   // center-left
        vec2( offset,  0.0f)   // center-right
    );

    const float gamma = 2.2;
	const bool edges = true;
void main()
{             
	vec3 col;
	if (edges){
		vec3 texSamples[5];
		for(int i = 0; i < 5; ++i){
			texSamples[i] = vec3(texture(forwardContrib, TexCoords.st + offsets[i]));
			texSamples[i] = texSamples[i].x > 0 && texSamples[i].y > 0 && texSamples[i].z > 0 ? texSamples[i] : texture(deferredContrib, TexCoords.st + offsets[i]).rgb;
		}	

		col = 4*texSamples[0];
		for(int i = 1; i < 5; i++)
			col += -texSamples[i];

	} else {
		col = vec3(texture(forwardContrib, TexCoords.st));
		col = col.x > 0 && col.y > 0 && col.z > 0 ? col : texture(deferredContrib, TexCoords.st).rgb;
	}
	
	vec3 result = vec3(1.0) - exp(-col * exposure);
    result = pow(result, vec3(1.0 / gamma));
    FragColor = vec4(result, 1.0);

}