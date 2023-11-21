#include "noise.h"

//--------------------------------------------------------------
void Noise::setup(int _width, int _height){
	ofDisableArbTex();
	shader.load("shader");

	float noiseFactor = 0.05;

	width = _width;
	height = _height;

	int w = width * noiseFactor;
	int h = height * noiseFactor;

	img.allocate(w, h, OF_IMAGE_GRAYSCALE);

	plane.set(width, height, w, h);
	plane.mapTexCoordsFromTexture(img.getTexture());
}

//--------------------------------------------------------------
void Noise::update(){
	float noiseScale = 0.1;
	float noiseVel = ofGetElapsedTimef() / 10.;

	ofPixels & pixels = img.getPixels();
	int w = img.getWidth();
	int h = img.getHeight();
	for(int y=0; y<h; y++) {
		for(int x=0; x<w; x++) {
			int i = y * w + x;
			float noiseVelue = ofNoise(x * noiseScale, y * noiseScale, noiseVel);
			pixels[i] = 255 * noiseVelue;
		}
	}
	img.update();
}

//--------------------------------------------------------------
void Noise::draw(){

	// bind our texture. in our shader this will now be tex0 by default
	// so we can just go ahead and access it there.
	img.getTexture().bind();

	shader.begin();

	ofPushMatrix();

	// translate plane into center screen.
	float tx = width / 2;
	float ty = height / 2;
	ofTranslate(tx, 0, ty);

	float rotation = 90.;
	ofRotateDeg(rotation, 1, 0, 0);

	plane.draw();

	ofPopMatrix();

	shader.end();
}
