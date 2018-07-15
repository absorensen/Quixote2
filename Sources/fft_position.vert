varying vec2 position;
uniform vec2 size;
void main()
{
	position = gl_Vertex.xy + size*0.5;
    gl_Position = ftransform();
}