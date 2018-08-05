// based on an implementation by Jeppe Revall Frisvad
// url: http://www2.compute.dtu.dk/pubdb/views/publication_details.php?id=5771
uniform sampler2D source;
void main()
{
  float g = texture2D(source, gl_TexCoord[0].xy).g;
  gl_FragColor.rgb = vec3(g, 0.0, 0.0);
}