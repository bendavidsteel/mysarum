#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	ofSetFrameRate(60);

	// constants
	display = 0;

	colourA = glm::vec3(1., 0., 0.);
	colourB = glm::vec3(0., 1., 0.);
	colourC = glm::vec3(1., 0., 0.);
	colourD = glm::vec3(0., 0., 1.);
	sunZ = 25;
	chemHeight = 1.;
	trailHeight = 2.;
	dayRate = 10.;

	feedMin = 0.01;
    feedRange = 0.09;

	// slime constants
	diffuseRate = 0.2;
	decayRate = 0.98;
	trailWeight = 1;

	// audio constants
	lowSmoothing = 0.6;
	highSmoothing = 0.8;

	// general constants
	sampleRate = 44100;
	bufferSize = 512;
    channels = 2;
	volume = 1.0;

	fileName = "testMovie";
    fileExt = ".mov"; // ffmpeg uses the extension to determine the container type. run 'ffmpeg -formats' to see supported formats

	// slime setup
	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = 0.04 * 2 * PI; // turnSpeed
	allSpecies[0].movementAttributes.z = CIRCLE; //spawn
	allSpecies[0].sensorAttributes.x = 30 * PI / 180; // sensorAngleRad
	allSpecies[0].sensorAttributes.y = 10; // sensorOffsetDist
	allSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);

	allSpecies[1].movementAttributes.x = 0.9; // moveSpeed
	allSpecies[1].movementAttributes.y = 0.08 * 2 * PI; // turnSpeed
	allSpecies[1].movementAttributes.z = RING;
	allSpecies[1].sensorAttributes.x = 40 * PI/ 180; // sensorAngleRad
	allSpecies[1].sensorAttributes.y = 20; //sensorOffsetDist
	allSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);
	newSpecies.resize(numSpecies);

	int numParticles = 1024 * 64;
	particles.resize(numParticles);

	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, ofGetWidth());
			p.pos.y = ofRandom(0, ofGetHeight());
		} else if (allSpecies[speciesIdx].movementAttributes.z == CIRCLE) {
			p.pos.x = ofGetWidth() / 2;
			p.pos.y = ofGetHeight() / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * ofGetWidth();
			p.pos.x = (ofGetWidth() / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (ofGetHeight() / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		}
		p.vel.x = ofRandom(-1, 1);
		p.vel.y = ofRandom(-1, 1);
		p.vel = glm::normalize(p.vel);
		p.vel = p.vel * allSpecies[speciesIdx].movementAttributes.x;
		p.attributes.x = speciesIdx;
	}
	
	particlesBuffer.allocate(particles, GL_DYNAMIC_DRAW);
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	ofPixels initialTrail;
	initialTrail.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);
	ofColor initialTrailColor(0., 0., 0., 0.);
	initialTrail.setColor(initialTrailColor);

	trailMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);
	trailMap.loadData(initialTrail);
	trailMap.bindAsImage(3, GL_READ_WRITE);

	// reaction diffusion setup
	reactionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	reactionMap.bindAsImage(0, GL_READ_WRITE);
	bool keepPattern = false;
	reSpawnReaction(keepPattern);

	// feedkillFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA16);

	// general setup
	flowFbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);

	// load slime shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_decay.glsl");
	compute_decay.linkProgram();

	compute_diffuse.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_diffuse.glsl");
	compute_diffuse.linkProgram();

	// load reaction diffusion shaders
	compute_feedkill.load("generic.vert", "compute_feedkill.frag");

	compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();

	// load general shaders
	compute_flow.load("generic.vert", "compute_flow.frag");

	renderer.load("generic.vert", "renderer.frag");
	simple_renderer.load("generic.vert", "simple_renderer.frag");

	// video recording
	ofAddListener(vidRecorder.outputFileCompleteEvent, this, &ofApp::recordingComplete);

	bRecording = false;

	pixels.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);

	// sound
	ofSoundStreamSettings settings;

	auto devices = soundStream.getDeviceList();
	if(!devices.empty()){
		settings.setInDevice(devices[1]);
	}

	settings.setInListener(this);
	settings.sampleRate = sampleRate;
	#ifdef TARGET_EMSCRIPTEN
		settings.numOutputChannels = 2;
	#else
		settings.numOutputChannels = 0;
	#endif
	settings.numInputChannels = channels;
	settings.bufferSize = bufferSize;
	soundStream.setup(settings);

	// audio analysis setup
	audioAnalyzer.setup(sampleRate, bufferSize, channels);

	int numBands = 24;
	vector<Component> melBands(numBands);
	for (int i = 0; i < numBands; i++) {
		melBands[i].value.x = float(i) / numBands;
	}
	melBandsBuffer.allocate(melBands, GL_DYNAMIC_DRAW);
	melBandsBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 5);

	int maxPoints = 4;
	points.resize(maxPoints);
	for (int i = 0; i < 2; i++) {
		points[i].value.x = 1;
		points[i].value.y = float(i) / 2;
		points[i].value.z = i;
		points[i].value.w = 1 - i;
	}
	for (int i = 2; i < 4; i++) {
		points[i].value.x = 0;
		points[i].value.y = 0;
		points[i].value.z = 0;
		points[i].value.w = 0;
	}
	newPoints.resize(maxPoints);
	
	pointsBuffer.allocate(points, GL_DYNAMIC_DRAW);
	pointsBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 6);

	compute_audio.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_audio.glsl");
	compute_audio.linkProgram();

	audioTexture.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	audioTexture.bindAsImage(7, GL_READ_WRITE);

	// video and optical flow setup
	vidGrabber.setVerbose(true);
	int sourceWidth = ofGetWidth();
	int sourceHeight = ofGetHeight();
	vidGrabber.setup(sourceWidth, sourceHeight);
	
	blurAmount = 15;
	cvDownScale = 10;
	bContrastStretch = false;
	// store a minimum squared value to apply flow velocity
	minLengthSquared = 0.5 * 0.5;

	int scaledWidth = sourceWidth / cvDownScale;
	int scaledHeight = sourceHeight / cvDownScale;

	currentImage.clear();
	// allocate the ofxCvGrayscaleImage currentImage
	currentImage.allocate(scaledWidth, scaledHeight);
	currentImage.set(0);
	// free up the previous cv::Mat
	previousMat.release();
	// copy the contents of the currentImage to previousMat
	// this will also allocate the previousMat
	currentImage.getCvMat().copyTo(previousMat);
	// free up the flow cv::Mat
	flowMat.release();
	// notice that the argument order is height and then width
	// store as floats
	flowMat = cv::Mat(scaledHeight, scaledWidth, CV_32FC2);

	opticalFlowPixels.allocate(scaledWidth, scaledHeight, OF_IMAGE_COLOR);
	optFlowTexture.allocate(scaledWidth, scaledHeight, GL_RG16);
	optFlowTexture.bindAsImage(4, GL_READ_WRITE);

	// setup midi
	// open port by number (you may need to change this)
	midiIn.openPort(1);
	//midiIn.openPort("IAC Pure Data In");	// by name
	//midiIn.openVirtualPort("ofxMidiIn Input"); // open a virtual port

	// don't ignore sysex, timing, & active sense messages,
	// these are ignored by default
	midiIn.ignoreTypes(false, false, false);

	// add ofApp as a listener
	midiIn.addListener(this);

	copyVariables();
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();

	float time = ofGetElapsedTimef();
	days = time / dayRate;
	time_of_day = fmod(days, float(2 * PI)) - PI;

	// video and optical flow
	vidGrabber.update();
	bool bNewFrame = vidGrabber.isFrameNew();
	
	if (bNewFrame){

		colorImg.setFromPixels(vidGrabber.getPixels());
		grayImage = colorImg;
		
		// flip the image horizontally
		grayImage.mirror(false, true);
		
		// scale down the grayImage into the smaller sized currentImage
		currentImage.scaleIntoMe(grayImage);
		
		if(bContrastStretch) {
			currentImage.contrastStretch();
		}
		
		if(blurAmount > 0 ) {
			currentImage.blurGaussian(blurAmount);
		}
		
		// to perform the optical flow, we will be using cv::Mat
		// so grab the cv::Mat from the current image and store in currentMat
		cv::Mat currentMat = currentImage.getCvMat();
		// calculate the optical flow
		// we pass in the previous mat, the curent one and the flowMat where the opti flow data will be stored
		cv::calcOpticalFlowFarneback(previousMat,
									 currentMat,
									 flowMat,
									 0.5, // pyr_scale
									 3, // levels
									 15, // winsize
									 3, // iterations
									 7, // poly_n
									 1.5, // poly_sigma
									 cv::OPTFLOW_FARNEBACK_GAUSSIAN);
		
		// copy over the current mat into the previous mat
		// so that the optical flow function can calculate the difference
		currentMat.copyTo(previousMat);
	}

	int numCols = ofGetWidth() / cvDownScale;
	int numRows = ofGetHeight() / cvDownScale;
	
	for( int x = 0; x < numCols; x++ ) {
		for( int y = 0; y < numRows; y++ ) {
			const cv::Point2f& fxy = flowMat.at<cv::Point2f>(y, x);
			glm::vec2 flowVector( fxy.x, fxy.y );
			if( glm::length2(flowVector) > minLengthSquared ) {
				ofFloatColor color( 0.5 + 0.5 * ofClamp(flowVector.x, -1, 1), 0.5 + 0.5 * ofClamp(flowVector.y, -1, 1), 0 );
				opticalFlowPixels.setColor(x, y, color);
			} else {
				opticalFlowPixels.setColor(x, y, ofFloatColor(0.5,0.5,0));
			}
		}
	}
	optFlowTexture.loadData(opticalFlowPixels);

	// audio analysis
	bool normalize = true;
	float rms = audioAnalyzer.getValue(RMS, 0, highSmoothing, normalize);
	vector<float> melBands = audioAnalyzer.getValues(MEL_BANDS, 0, lowSmoothing);

	bool isOnset = audioAnalyzer.getOnsetValue(0);

	int numBands = 24;
	vector<Component> melBandsComponents(numBands);
	for(int i = 0; i < numBands; i++){
		melBandsComponents[i].value.x = ofMap(melBands[i], DB_MIN, DB_MAX, 0.0, 1.0, true);//clamped value
	}
	melBandsBuffer.updateData(melBandsComponents);

	moveToVariables();
	vector<Component> thesePoints(points.size());
	float a = pow(rms, 3) / 2;
	for (int i = 0; i < points.size(); i++) {
		thesePoints[i].value.x = a * points[i].value.x * cos((2 * PI * points[i].value.y) + time_of_day);
		thesePoints[i].value.y = a * points[i].value.x * sin((2 * PI * points[i].value.y) + time_of_day);
		thesePoints[i].value.z = points[i].value.z;
		thesePoints[i].value.w = points[i].value.w;
	}
	pointsBuffer.updateData(thesePoints);

	if (bReSpawnAgents) {
		reSpawnAgents();
		bReSpawnAgents = false;
	}

	if (bReSpawnReaction) {
		bool keepPattern = true;
		reSpawnReaction(keepPattern);
		bReSpawnReaction = false;
	}

	int workGroupSize = 20;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	// general updates
	flowFbo.begin();
	ofClear(255,255,255, 0);
	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	ofSetColor(255);
	ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	compute_flow.end();
	flowFbo.end();

	compute_audio.begin();
	compute_audio.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_audio.setUniform1f("deltaTime", deltaTime);
	compute_audio.setUniform1i("numBands", numBands);
	compute_audio.setUniform1f("angle", time_of_day);
	compute_audio.setUniform1f("rms", rms);
	compute_audio.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_audio.end();

	// slime updates
	// horizontal blur
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2i("blurDir", 1, 0);
	compute_diffuse.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_diffuse.end();

	// vertical blur
	compute_diffuse.begin();
	compute_diffuse.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform2i("blurDir", 0, 1);
	compute_diffuse.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_diffuse.end();

	compute_decay.begin();
	compute_decay.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_decay.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_decay.end();

	compute_agents.begin();
	compute_agents.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", time);
	compute_agents.setUniform1f("trailWeight", trailWeight);
	compute_agents.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_agents.setUniform1f("agentFlowMag", agentFlowMag);
	compute_agents.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
	
	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of particles be < 1024
	compute_agents.dispatchCompute((particles.size() + 1024 -1 )/1024, 1, 1);
	compute_agents.end();

	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_reaction.setUniform1f("reactionFlowMag", reactionFlowMag);
	compute_reaction.setUniform1f("feedMin", feedMin);
	compute_reaction.setUniform1f("feedRange", feedRange);
	compute_reaction.setUniformTexture("flowMap", flowFbo.getTexture(), 2);
	compute_reaction.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_reaction.end();
}

//--------------------------------------------------------------
void ofApp::draw() {

	float sun_x = (ofGetWidth() / 2) + (2 * ofGetWidth() / 3) * cos(time_of_day);
	float sun_y = (ofGetHeight() / 2) + (2 * ofGetHeight() / 3) * sin(time_of_day);

	if (display == 0) {
		fbo.begin();
		ofClear(255,255,255, 0);
		renderer.begin();
		renderer.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
		renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
		renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
		renderer.setUniform3f("colourC", colourC.x, colourC.y, colourC.z);
		renderer.setUniform3f("colourD", colourD.x, colourD.y, colourD.z);
		renderer.setUniform3f("light", sun_x, sun_y, sunZ);
		renderer.setUniform1f("chem_height", chemHeight);
		renderer.setUniform1f("trail_height", trailHeight);
		renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
		ofSetColor(255);
		ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
		renderer.end();
		fbo.end();
		fbo.draw(0, 0);
	}  else if (display == 1 || display == 2 || display == 3 || display == 4){
		fbo.begin();
		ofClear(255,255,255, 0);
		simple_renderer.begin();
		simple_renderer.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
		simple_renderer.setUniform3f("colourA", colourA.x, colourA.y, colourA.z);
		simple_renderer.setUniform3f("colourB", colourB.x, colourB.y, colourB.z);
		simple_renderer.setUniform3f("colourC", colourC.x, colourC.y, colourC.z);
		simple_renderer.setUniform3f("colourD", colourD.x, colourD.y, colourD.z);
		simple_renderer.setUniform3f("light", sun_x, sun_y, sunZ);
		simple_renderer.setUniform1f("chem_height", chemHeight);
		simple_renderer.setUniform1f("trail_height", trailHeight);
		simple_renderer.setUniform1i("opticalFlowDownScale", cvDownScale);
		simple_renderer.setUniform1i("display", display);
		simple_renderer.setUniformTexture("flowMap", flowFbo.getTexture(), 0);
		ofSetColor(255);
		ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
		simple_renderer.end();
		fbo.end();
		fbo.draw(0, 0);
	} else if (display == 5) {
		flowFbo.draw(0, 0, ofGetWidth(), ofGetHeight());
	}

	if(bRecording){
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
	reactionMap.clear();

	soundStream.close();

	midiIn.closePort();
	midiIn.removeListener(this);
}

//--------------------------------------------------------------
void ofApp::audioIn(ofSoundBuffer & buffer){
	if (buffer.getNumChannels() == audioAnalyzer.getChannelAnalyzersPtrs().size()) {
		buffer *= volume;
		audioAnalyzer.analyze(buffer);
	}
	
	if(bRecording)
        vidRecorder.addAudioSamples(buffer.getBuffer().data(), bufferSize, channels);
}

//--------------------------------------------------------------
void ofApp::recordingComplete(ofxVideoRecorderOutputFileCompleteEventArgs& args){
    cout << "The recoded video file is now complete." << endl;
}

//--------------------------------------------------------------
void ofApp::newMidiMessage(ofxMidiMessage& message) {

	if (message.status < MIDI_SYSEX) {
		if (message.status == MIDI_NOTE_ON) {
			// 36 - 96
			newInput(message.pitch);
		} else if (message.status == MIDI_CONTROL_CHANGE) {
			int val = message.value;
			// 0 - 127
			newReactionFlowMag = ofMap(val, 0, 127, 0., 30.);
		} else if (message.status == MIDI_PITCH_BEND) {
			int val = message.value;
			// 0 - 16383
			if (val > 8192) {
				newAgentFlowMag = ofMap(val, 8192, 16383, 0., 30.);
			} else {
				newAgentFlowMag = 0.;
			}
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == '1') {
		newInput(36);
	} else if (key == '2') {
		newInput(37);
	} else if (key == '3') {
		newInput(38);
	} else if (key == '4') {
		newInput(39);
	} else if (key == '5') {
		newInput(40);
	} else if (key == 'q') {
		newInput(43);
	} else if (key == 'w') {
		newInput(44);
	} else if (key == 'e') {
		newInput(45);
	} else if (key == 'r') {
		newInput(49);
	} else if (key == 't') {
		newInput(50);
	} else if (key == 'a') {
		newInput(62);
	} else if (key == 's') {
		newInput(63);
	} else if (key == 'z') {
		newInput(66);
	} else if (key == 'x') {
		newInput(67);
	} else if (key == 'c') {
		newInput(68);
	} else if (key == 'v') {
		newInput(69);
	} else if (key == 'b') {
		newInput(70);
	}
}

void ofApp::newInput(int key) {
	
	if (key == 36) {
		bReSpawnAgents = true;
	} else if (key == 37) {
		display = 0;
	} else if (key == 38) {
		display = 1;
	} else if (key == 39) {
		display = 2;
	} else if (key == 40) {
		display = 3;
	} else if (key == 41) {
		display = 4;
	} else if (key == 42) {
		display = 5;
	} else if (key == 43) {
		newColourA = glm::vec3(1., 0., 0.);
		newColourB = glm::vec3(0., 1., 0.);
	} else if (key == 44) {
		newColourA = glm::vec3(1.0, 0.906, 0.51);
		newColourB = glm::vec3(0.98, 0.345, 0.118);
	} else if (key == 45) {
		newColourA = glm::vec3(0.494, 0.921, 0.063);
		newColourB = glm::vec3(0.839, 0.812, 0.153);
	} else if (key == 46) {
		newColourC = glm::vec3(0.839, 0.02, 0.004);
		newColourD = glm::vec3(0., 0., 1.);
	} else if (key == 47) {
		newColourC = glm::vec3(1., 0., 0.);
		newColourD = glm::vec3(0., 0., 1.);
	} else if (key == 48) {
		newColourC = glm::vec3(191./255., 11./255., 59./255.);
		newColourD = glm::vec3(213./255., 13./255., 216./255.);
	} else if (key == 49) {
		newSpecies[0].colour = glm::vec4(0.796, 0.2, 1., 1.);
		newSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);
	} else if (key == 50) {
		newSpecies[0].colour = glm::vec4(0.263, 0.31, 0.98, 1.);
		newSpecies[1].colour = glm::vec4(0.396, 0.839, 0.749, 1.);
	} else if (key == 51) {
		newSpecies[0].movementAttributes.x = ofRandom(0.9, 1.4); // moveSpeed
	} else if (key == 52) {
		newSpecies[0].movementAttributes.y = ofRandom(0.03, 0.07) * 2 * PI; // turnSpeed
	} else if (key == 53) {
		newSpecies[0].sensorAttributes.x = ofRandom(30, 50) * PI / 180; // sensorAngleRad
	} else if (key == 54) {
		newSpecies[0].sensorAttributes.y = ofRandom(10, 30); // sensorOffsetDist
	} else if (key == 55) {
		newSpecies[1].movementAttributes.x = ofRandom(0.6, 1.1); // moveSpeed
	} else if (key == 56) {
		newSpecies[1].movementAttributes.y = ofRandom(0.08, 0.13) * 2 * PI; // turnSpeed
	} else if (key == 57) {
		newSpecies[1].sensorAttributes.x = ofRandom(50, 70) * PI/ 180; // sensorAngleRad
	} else if (key == 58) {
		newSpecies[1].sensorAttributes.y = ofRandom(30, 50); //sensorOffsetDist
	} else if (key == 59) {
		newTrailHeight = ofRandom(5, 10);
	} else if (key == 60) {
		newDiffuseRate = ofRandom(0.05, 0.4);
	} else if (key == 61) {
		newDecayRate = 1 - std::pow(0.1, ofRandom(1, 3.1));
	} else if (key == 62) {
		if (newAgentFlowMag == 0) {
			newAgentFlowMag = 20;
		} else if (newAgentFlowMag = 20) {
			newAgentFlowMag = 0;
		}
	} else if (key == 63) {
		if (newReactionFlowMag == 0) {
			newReactionFlowMag = 20;
		} else if (newReactionFlowMag = 20) {
			newReactionFlowMag = 0;
		}
	} else if (key == 64) {
		newSunZ = ofRandom(20, 50);
	} else if (key == 65) {
		newChemHeight = ofRandom(1, 9);
	} else if (key == 66) {
		newFeedMin = 0.01;
		newFeedRange = 0.09;
		bReSpawnReaction = true;
	} else if (key == 67) {
		newFeedMin = 0.01;
		newFeedRange = 0.025;
		bReSpawnReaction = true;
	} else if (key == 68) {
		newFeedMin = 0.035;
		newFeedRange = 0.025;
		bReSpawnReaction = true;
	} else if (key == 69) {
		newFeedMin = 0.06;
		newFeedRange = 0.015;
		bReSpawnReaction = true;
	} else if (key == 70) {
		newFeedMin = 0.075;
		newFeedRange = 0.015;
		bReSpawnReaction = true;
	} else if (key == 71) {
		for (int i = 0; i < 1; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0;
			newPoints[i].value.y = 0;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 1; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 72) {
		for (int i = 0; i < 2; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 2;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 2; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 73) {
		for (int i = 0; i < 2; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.5 + 0.5 * i;
			newPoints[i].value.y = float(i) / 2;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 2; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 74) {
		for (int i = 0; i < 2; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.5 + 0.5 * i;
			newPoints[i].value.y = float(i) / 3;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 2; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 75) {
		for (int i = 0; i < 3; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 3;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 76) {
		for (int i = 0; i < 3; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 3.5;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 77) {
		for (int i = 0; i < 3; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.5 + i * 0.5;
			newPoints[i].value.y = float(i) / 3.5;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
	} else if (key == 78) {
		for (int i = 0; i < 4; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 4;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		newPoints[4].value = glm::vec4(0);
	} else if (key == 79) {
		for (int i = 0; i < 4; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 9;
			newPoints[i].value.z = z;
			newPoints[i].value.w =  1 - z;
		}
		newPoints[4].value = glm::vec4(0);
	} else if (key == 80) {
		for (int i = 0; i < 4; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.3 + 0.3 * i;
			newPoints[i].value.y = float(i) / 4;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
		newPoints[4].value = glm::vec4(0);
	} else if (key == 81) {
		for (int i = 0; i < 5; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 5;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
	} else if (key == 82) {
		for (int i = 0; i < 5; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.2 + 0.2 * i;
			newPoints[i].value.y = float(i) / 8;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
	} else if (key == 83) {
		for (int i = 0; i < 5; i++) {
			float z = std::round(ofRandom(0, 1.1));
			newPoints[i].value.x = 0.3 + 0.3 * i;
			newPoints[i].value.y = float(i) / 6;
			newPoints[i].value.z = z;
			newPoints[i].value.w = 1 - z;
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	if(key=='['){
		bRecording = !bRecording;
		if(bRecording && !vidRecorder.isInitialized()) {
			vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, ofGetWidth(), ofGetHeight(), 60, sampleRate, channels);
		    // vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, vidGrabber.getWidth(), vidGrabber.getHeight(), 30); // no audio
		    // vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, 0,0,0, sampleRate, channels); // no video
		    // vidRecorder.setupCustomOutput(vidGrabber.getWidth(), vidGrabber.getHeight(), 30, sampleRate, channels, "-vcodec mpeg4 -b 1600k -acodec mp2 -ab 128k -f mpegts udp://localhost:1234"); // for custom ffmpeg output string (streaming, etc)

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
	if(key==']'){
		bRecording = false;
		vidRecorder.close();
	}
}


void ofApp::copyVariables() {
	for (int i = 0; i < points.size(); i++) {
		newPoints[i].value.x = points[i].value.x;
		newPoints[i].value.y = points[i].value.y;
		newPoints[i].value.z = points[i].value.z;
		newPoints[i].value.w = points[i].value.w;
	}

	for (int i = 0; i < 2; i++) {
		newSpecies[i].movementAttributes.x = allSpecies[i].movementAttributes.x;
		newSpecies[i].movementAttributes.y = allSpecies[i].movementAttributes.y;
		newSpecies[i].movementAttributes.z = allSpecies[i].movementAttributes.z;
		newSpecies[i].sensorAttributes.x = allSpecies[i].sensorAttributes.x;
		newSpecies[i].sensorAttributes.y = allSpecies[i].sensorAttributes.y;
		newSpecies[i].colour = allSpecies[i].colour;
	}

	newColourA = colourA;
	newColourB = colourB;
	newColourC = colourC;
	newColourD = colourD;

	newSunZ = sunZ;
	newDayRate = dayRate;
	newChemHeight = chemHeight;
	newTrailHeight = trailHeight;

	newDiffuseRate = diffuseRate;
	newDecayRate = decayRate;
	newTrailWeight = trailWeight;

	newFeedMin = feedMin;
	newFeedRange = feedRange;

	newReactionFlowMag = reactionFlowMag;
	newAgentFlowMag = agentFlowMag;
}

void ofApp::moveToVariables() {
	float rate = 0.02;

	for (int i = 0; i < points.size(); i++) {
		points[i].value = glm::mix(points[i].value, newPoints[i].value, rate);
	}

	for (int i = 0; i < 2; i++) {
		allSpecies[i].movementAttributes.x = (1 - rate) * allSpecies[i].movementAttributes.x + rate * newSpecies[i].movementAttributes.x;
		allSpecies[i].movementAttributes.y = (1 - rate) * allSpecies[i].movementAttributes.y + rate * newSpecies[i].movementAttributes.y;
		allSpecies[i].sensorAttributes = glm::mix(allSpecies[i].sensorAttributes, newSpecies[i].sensorAttributes, rate);
		allSpecies[i].colour = glm::mix(allSpecies[i].colour, newSpecies[i].colour, rate);
	}
	allSpeciesBuffer.updateData(allSpecies);

	colourA = glm::mix(colourA, newColourA, rate);
	colourB = glm::mix(colourB, newColourB, rate);
	colourC = glm::mix(colourC, newColourC, rate);
	colourD = glm::mix(colourD, newColourD, rate);

	sunZ = (1 - rate) * sunZ + rate * newSunZ;
	chemHeight = (1 - rate) * chemHeight + rate * newChemHeight;
	trailHeight = (1 - rate) * trailHeight + rate * newTrailHeight;

	diffuseRate = (1 - rate) * diffuseRate + rate * newDiffuseRate;
	decayRate = (1 - rate) * decayRate + rate * newDecayRate;
	trailWeight = (1 - rate) * trailWeight + rate * newTrailWeight;

	feedMin = (1 - rate) * feedMin + rate * newFeedMin;
	feedRange = (1 - rate) * feedRange + rate * newFeedRange;

	reactionFlowMag = (1 - rate) * reactionFlowMag + rate * newReactionFlowMag;
	agentFlowMag = (1 - rate) * agentFlowMag + rate * newAgentFlowMag;

	dayRate = (1 - rate) * dayRate + rate * newDayRate;
}

void ofApp::reSpawnAgents() {
	allSpecies[0].movementAttributes.z = std::round(ofRandom(0, 5.1));
	allSpecies[1].movementAttributes.z = std::round(ofRandom(0, 5.1));
	int numSpecies = 2;
	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, ofGetWidth());
			p.pos.y = ofRandom(0, ofGetHeight());
		} else if (allSpecies[speciesIdx].movementAttributes.z == CIRCLE) {
			p.pos.x = ofGetWidth() / 2;
			p.pos.y = ofGetHeight() / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * ofGetWidth();
			p.pos.x = (ofGetWidth() / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (ofGetHeight() / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		} else if (allSpecies[speciesIdx].movementAttributes.z == SMALL_RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.15 * ofGetWidth();
			p.pos.x = (ofGetWidth() / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (ofGetHeight() / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
		} else if (allSpecies[speciesIdx].movementAttributes.z == VERTICAL_LINE) {
			p.pos.x = (ofGetWidth() / 2) * ofRandom(0.99, 1.01);
			p.pos.y = ofRandom(0, ofGetHeight());
		} else if (allSpecies[speciesIdx].movementAttributes.z == HORIZONTAL_LINE) {
			p.pos.x = ofRandom(0, ofGetWidth());
			p.pos.y = (ofGetHeight() / 2) * ofRandom(0.99, 1.01);
		}
		p.vel.x = ofRandom(-1, 1);
		p.vel.y = ofRandom(-1, 1);
		p.vel = glm::normalize(p.vel);
		p.vel = p.vel * allSpecies[speciesIdx].movementAttributes.x;
		p.attributes.x = speciesIdx;
	}
	particlesBuffer.updateData(particles);
}

void ofApp::reSpawnReaction(bool keepPattern) {
	ofPixels initialReaction;
	initialReaction.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);

	if (keepPattern) {
		reactionMap.readToPixels(initialReaction);
	} else {
		ofColor baseReactionColour(255., 0., 0., 0.);
		initialReaction.setColor(baseReactionColour);
	}
	
	ofColor reactantColour(255., 255., 0., 0.);
	for (int j = 0; j < ofGetHeight(); j++) {
		for (int i = 0; i < ofGetWidth(); i++) {
			if (ofRandom(1) < 0.1) {
				initialReaction.setColor(i, j, reactantColour);
			}
		}
	}

	reactionMap.loadData(initialReaction);
}
