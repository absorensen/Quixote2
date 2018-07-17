// based on an implementation by Jeppe Revall Frisvad
// url: http://www2.compute.dtu.dk/pubdb/views/publication_details.php?id=5771

varying vec2 position;
uniform vec2 size;
void main()
{
	position = gl_Vertex.xy + size*0.5;
    gl_Position = ftransform();
}