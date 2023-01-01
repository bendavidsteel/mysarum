#include "ofMain.h"
#include "ofApp.h"

const int screenHeight = 768;
const int screenWidth = 1024;

//========================================================================
int main( ){
	ofSetupOpenGL(screenWidth, screenHeight, OF_WINDOW);			// <-------- setup the GL context

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());

}
