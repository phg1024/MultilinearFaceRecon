void main() {
	gl_FragColor = vec4 (gl_Color.xyz, gl_FragCoord.z);
}