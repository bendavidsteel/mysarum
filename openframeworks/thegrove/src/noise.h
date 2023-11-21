#pragma once

#include "ofMain.h"

class Noise{
	public:
		
	void setup(int width, int height);
	void update();
	void draw();

	ofShader shader;
	ofPlanePrimitive plane;
	ofImage img;

	int width, height;
};
