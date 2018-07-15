varying vec2 position;
void main()
{
    position = gl_Vertex.xy;
	gl_Position = ftransform();
}