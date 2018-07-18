uniform sampler2D source;
void main()
{
  float g = texture2D(source, gl_TexCoord[0].xy).g;
  gl_FragColor.rgb = vec3(g, 0.0, 0.0);
}