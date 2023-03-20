#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	volDepth = 10;

	diffuseRate = 0.05;
	decayRate = 0.98;
	trailWeight = 1;

	sampleRate = 44100;
    channels = 2;

	fileName = "testMovie";
    fileExt = ".mov"; // ffmpeg uses the extension to determine the container type. run 'ffmpeg -formats' to see supported formats

	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	float angle = 30 * PI / 180;
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = 1.0; // turnStrength
	allSpecies[0].movementAttributes.z = CENTRE; //spawn
	allSpecies[0].sensorAttributes.x = 30; // sensorDist
	allSpecies[0].sensorAttributes.y = allSpecies[0].sensorAttributes.x * sin(angle); // sensorOffset
	allSpecies[0].sensorAttributes.z = allSpecies[0].sensorAttributes.y / tan(angle); // sensorOffDist
	allSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);

	angle = 60 * PI / 180;
	allSpecies[1].movementAttributes.x = 0.9;
	allSpecies[1].movementAttributes.y = 0.6;
	allSpecies[1].movementAttributes.z = RING;
	allSpecies[1].sensorAttributes.x = 40;
	allSpecies[1].sensorAttributes.y = allSpecies[1].sensorAttributes.x * sin(angle);
	allSpecies[1].sensorAttributes.z = allSpecies[1].sensorAttributes.y / tan(angle);
	allSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);

	int numParticles = 1024 * 1;
	particles.resize(numParticles);

	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, volWidth);
			p.pos.y = ofRandom(0, volHeight);
			p.pos.z = ofRandom(0, volDepth);
		} else if (allSpecies[speciesIdx].movementAttributes.z == CENTRE) {
			p.pos.x = volWidth / 2;
			p.pos.y = volHeight / 2;
			p.pos.z = volDepth / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * volWidth;
			p.pos.x = (volWidth / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (volHeight / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
			p.pos.z = ofRandom(0, volDepth);
		}
		p.vel.x = ofRandom(-1, 1);
		p.vel.y = ofRandom(-1, 1);
		p.vel.z = ofRandom(-1, 1);
		p.vel = glm::normalize(p.vel);
		p.vel = p.vel * allSpecies[speciesIdx].movementAttributes.x;
		p.attributes.x = speciesIdx;
	}
	
	particlesBuffer.allocate(particles, GL_DYNAMIC_DRAW);
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	ofPixels initialTrail;
	initialTrail.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);
	ofColor initialTrailColor(0., 0., 0., 0.);
	initialTrail.setColor(initialTrailColor);

	trailMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);
	trailMap.loadData(initialTrail);
	trailMap.bindAsImage(2, GL_READ_WRITE);

	flowMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);
	flowMap.bindAsImage(3, GL_READ_WRITE);

	// load shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_diffuse_decay.glsl");
	compute_decay.linkProgram();

	compute_flow_field.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_flow_field.glsl");
	compute_flow_field.linkProgram();

	renderer.load("generic.vert", "shader.frag");

	// video recording
	ofAddListener(vidRecorder.outputFileCompleteEvent, this, &ofApp::recordingComplete);

	bRecording = false;

	// sound
	ofSoundStreamSetup(0, 2, 44100, 256, 4);
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();

	compute_flow_field.begin();
	compute_flow_field.dispatchCompute(ofGetWidth()/)

	compute_decay.begin();
	compute_decay.setUniform3i("resolution", ofGetWidth(), ofGetHeight(), volDepth);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("time", ofGetElapsedTimef());
	compute_decay.setUniform1f("diffuseRate", diffuseRate);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.dispatchCompute(ofGetWidth()/32, ofGetHeight()/32, 1);
	compute_decay.end();

	compute_agents.begin();
	compute_agents.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", ofGetElapsedTimef());
	compute_agents.setUniform1f("trailWeight", trailWeight);
	
	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of particles be < 1024
	compute_agents.dispatchCompute((particles.size() + 1024 -1 )/1024, 1, 1);
	compute_agents.end();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fbo.begin();
	renderer.begin();
	renderer.setUniform2i("screen_res", ofGetWidth(), ofGetHeight());
	renderer.setUniform2i("trail_res", volWidth, volHeight);
	ofSetColor(255);
	ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	renderer.end();
	fbo.end();
	fbo.draw(0, 0);

	if(bRecording){
		// const ofFbo fbo = volume.getFbo();
		fbo.readToPixels(pixels);
		bool recordSuccess = vidRecorder.addFrame(pixels);
		if (!recordSuccess) {
			ofLogWarning("This frame was not added!");
		}
	}

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

	trailMap.clear();
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