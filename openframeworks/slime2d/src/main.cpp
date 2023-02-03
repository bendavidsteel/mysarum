#include "ofMain.h"
#include "ofApp.h"

const int screenHeight = 768;
const int screenWidth = 1024;

//========================================================================
int main( ){
	ofGLWindowSettings settings;
	settings.setGLVersion(4,3);
	settings.windowMode = OF_WINDOW;
	settings.setSize(screenWidth, screenHeight);
	ofCreateWindow(settings);			// <-------- setup the GL context

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());

}
