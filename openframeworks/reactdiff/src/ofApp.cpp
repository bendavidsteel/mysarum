#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	sampleRate = 44100;
    channels = 2;

	float_strength = 1;

	ofPixels initialReaction;
	initialReaction.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);
	ofColor baseReactionColour(255., 0., 0., 0.);
	initialReaction.setColor(baseReactionColour);

	InitialPattern initialPattern = RANDOM;

	ofColor reactantColour(255., 255., 0., 0.);
	if (initialPattern == CIRCLE) {
		int circleSize = ofGetHeight() / 32;
		glm::vec2 centre = glm::vec2(ofGetWidth() / 2, ofGetHeight() / 2);
		for (int j = 0; j < ofGetHeight(); j++) {
			for (int i = 0; i < ofGetWidth(); i++) {
				glm::vec2 point = glm::vec2(i, j);
				float radius = glm::distance(point, centre);
				if (radius < circleSize) {
					initialReaction.setColor(i, j, reactantColour);
				}
			}
		}
	} else if (initialPattern == RANDOM) {
		for (int j = 0; j < ofGetHeight(); j++) {
			for (int i = 0; i < ofGetWidth(); i++) {
				if (ofRandom(1) < 0.1) {
					initialReaction.setColor(i, j, reactantColour);
				}
			}
		}
	}

	reactionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	reactionMap.loadData(initialReaction);
	reactionMap.bindAsImage(0, GL_READ_WRITE);

	feedkillMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	feedkillMap.bindAsImage(1, GL_READ_WRITE);

	flowMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);
	flowMap.bindAsImage(2, GL_READ_WRITE);

	diffusionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	diffusionMap.bindAsImage(3, GL_READ_WRITE);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);

	// load shaders
	compute_flow.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_flow.glsl");
	compute_flow.linkProgram();

	compute_diffusion.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_diffusion.glsl");
	compute_diffusion.linkProgram();

	compute_feedkill.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_feedkill.glsl");
	compute_feedkill.linkProgram();

	compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();

	float planeScale = 0.75;
	int planeWidth = ofGetWidth() * planeScale;
	int planeHeight = ofGetHeight() * planeScale;
	int planeGridSize = 20;
	int planeColums = planeWidth / planeGridSize;
	int planeRows = planeHeight / planeGridSize;

	plane.set(planeWidth, planeHeight, planeColums, planeRows, OF_PRIMITIVE_TRIANGLES);

	renderer.load("renderer.vert", "renderer.frag");
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();
	int workGroupSize = 32;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_flow.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_flow.end();

	compute_diffusion.begin();
	compute_diffusion.setUniform1f("time", time);
	compute_diffusion.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_diffusion.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_diffusion.end();

	compute_feedkill.begin();
	compute_feedkill.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_feedkill.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_feedkill.end();

	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.setUniform1f("float_strength", float_strength);
	compute_reaction.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_reaction.end();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	// ofClear(255,255,255, 0);
	renderer.begin();
	renderer.setUniform3f("colourA", 1., 0., 0.);
	renderer.setUniform3f("colourB", 0., 1., 0.);
	renderer.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	renderer.setUniform3f("light", ofGetWidth(), 0., 10.);
	renderer.setUniform1f("height", 1000.);

	// translate plane into center screen.
	float tx = ofGetWidth() / 2;
	float ty = ofGetHeight() / 2;
	ofTranslate(tx, ty);

	plane.drawWireframe();

	renderer.end();
	fbo.end();
	fbo.draw(0, 0, ofGetWidth(), ofGetHeight());

    // Check if the video recorder encountered any error while writing video frame or audio smaples.
    if (vidRecorder.hasVideoError()) {
        ofLogWarning("The video recorder failed to write some frames!");
    }

    if (vidRecorder.hasAudioError()) {
        ofLogWarning("The video recorder failed to write some audio samples!");
    }

	ofDrawBitmapString(ofGetFrameRate(),20,20);
}

void ofApp::exit(){
    ofRemoveListener(vidRecorder.outputFileCompleteEvent, this, &ofApp::recordingComplete);
    vidRecorder.close();

	
}

//--------------------------------------------------------------
void ofApp::audioIn(float *input, int bufferSize, int nChannels){
    if(bRecording)
        vidRecorder.addAudioSamples(input, bufferSize, nChannels);
}

//--------------------------------------------------------------
void ofApp::recordingComplete(ofxVideoRecorderOutputFileCompleteEventArgs& args){
    cout << "The recoded video file is now complete." << endl;
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	if(key=='r'){
		bRecording = !bRecording;
		if(bRecording && !vidRecorder.isInitialized()) {
			vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, ofGetWidth(), ofGetHeight(), 30, sampleRate, channels);
		//          vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, vidGrabber.getWidth(), vidGrabber.getHeight(), 30); // no audio
		//            vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, 0,0,0, sampleRate, channels); // no video
		//          vidRecorder.setupCustomOutput(vidGrabber.getWidth(), vidGrabber.getHeight(), 30, sampleRate, channels, "-vcodec mpeg4 -b 1600k -acodec mp2 -ab 128k -f mpegts udp://localhost:1234"); // for custom ffmpeg output string (streaming, etc)

		// Start recording
			vidRecorder.start();
		}
		else if(!bRecording && vidRecorder.isInitialized()) {
			vidRecorder.setPaused(true);
		}
		else if(bRecording && vidRecorder.isInitialized()) {
			vidRecorder.setPaused(false);
		}
	}
	if(key=='c'){
		bRecording = false;
		vidRecorder.close();
	}
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}