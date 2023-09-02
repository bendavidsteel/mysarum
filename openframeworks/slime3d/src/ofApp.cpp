#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	volWidth = 512;
	volHeight = 512;
	volDepth = 32;

	diffuseRate = 0.3;
	decayRate = 0.96;
	trailWeight = 1;

	sampleRate = 44100;
    channels = 2;

	fileName = "testMovie";
    fileExt = ".mov"; // ffmpeg uses the extension to determine the container type. run 'ffmpeg -formats' to see supported formats

	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = 1.0; // turnStrength
	allSpecies[0].movementAttributes.z = CENTRE; //spawn
	allSpecies[0].sensorAttributes.x = 30 * PI / 180; // sensorAngleRad
	allSpecies[0].sensorAttributes.y = 30; // sensorOffsetDist
	allSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);

	allSpecies[1].movementAttributes.x = 0.9;
	allSpecies[1].movementAttributes.y = 0.6;
	allSpecies[1].movementAttributes.z = RING;
	allSpecies[1].sensorAttributes.x = 60 * PI / 180;
	allSpecies[1].sensorAttributes.y = 40;
	allSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);

	int numParticles = 1024 * 32;
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

	float * volumeData = new float[volWidth*volHeight*volDepth*4];
    for(int z=0; z<volDepth; z++)
    {
        for(int x=0; x<volWidth; x++)
        {
            for(int y=0; y<volHeight; y++)
            {
                // convert from greyscale to RGBA, false color
                int i4 = ((x+volWidth*y)+z*volWidth*volHeight)*4;
                ofColor c(0., 0., 0., 0.);

                volumeData[i4] = c.r;
                volumeData[i4+1] = c.g;
                volumeData[i4+2] = c.b;
                volumeData[i4+3] = c.a;
            }
        }
    }

	trailMap.allocate(volWidth, volHeight, volDepth, GL_RGBA8);
	trailMap.loadData(volumeData, volWidth, volHeight, volDepth, 0, 0, 0, GL_RGBA);
	trailMap.bindAsImage(2, GL_READ_WRITE);

	flowMap.allocate(volWidth, volHeight, volDepth, GL_RGBA8);
	flowMap.bindAsImage(3, GL_READ_WRITE);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);

	// load shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_decay.glsl");
	compute_decay.linkProgram();

	compute_diffuse.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_diffuse.glsl");
	compute_diffuse.linkProgram();

	compute_flow.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_flow.glsl");
	compute_flow.linkProgram();

	renderer.load("renderer.vert", "renderer.frag");

	// volume.setup(volWidth, volHeight, volDepth, ofVec3f(1,1,1),false);
	// volume.setRenderSettings(2.0, 0.1, 1.0, 0.0);

	// video recording
	ofAddListener(vidRecorder.outputFileCompleteEvent, this, &ofApp::recordingComplete);

	bRecording = false;

	// sound
	ofSoundStreamSetup(0, 2, 44100, 256, 4);
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	int localSizeX = 4;
	int localSizeY = 4;
	int localSizeZ = 4;

	// horizontal blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 1, 0, 0);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	compute_diffuse.end();

	// vertical blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 1, 0);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	compute_diffuse.end();

	// depth blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 0, 1);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	compute_diffuse.end();

	compute_decay.begin();
	compute_decay.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	compute_decay.end();

	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", volWidth, volHeight);
	compute_flow.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	compute_flow.end();

	compute_agents.begin();
	compute_agents.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", time);
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
	volume.drawVolume(0,0,0, ofGetHeight(), 0);

	// fbo.begin();
	// renderer.begin();
	// renderer.setUniform2i("screen_res", ofGetWidth(), ofGetHeight());
	// renderer.setUniform3i("trail_res", volWidth, volHeight, volDepth);
	// ofSetColor(255);
	// ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	// renderer.end();
	// fbo.end();
	// fbo.draw(0, 0);

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