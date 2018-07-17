// based on an implementation by Jeppe Revall Frisvad
// url: http://www2.compute.dtu.dk/pubdb/views/publication_details.php?id=5771

varying vec2 position;
void main()
{
    position = gl_Vertex.xy;
	gl_Position = ftransform();
}