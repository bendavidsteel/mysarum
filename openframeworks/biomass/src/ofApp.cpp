#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	ofSetFrameRate(120);

	// audio constants
	lowSmoothing = 0.4;
	highSmoothing = 0.6;

	// general constants
	sampleRate = 48000;
	bufferSize = 1024;
    channels = 2;
	volume = 1.0;

	biomass.setup();
	evolution.setup(biomass);

	// general setup
	int mapWidth = biomass.getMapWidth();
	int mapHeight = biomass.getMapHeight();
	fbo.allocate(mapWidth, mapHeight, GL_RGBA8);

	postprocess.load("generic.vert", "postprocess.frag");

	// sound
	ofSoundStreamSettings settings;

	// auto devices = soundStream.getDeviceList(ofSoundDevice::Api::ALSA);
	auto devices = soundStream.getDeviceList();
	if(!devices.empty()){
		settings.setInDevice(devices[0]);
	}
	// settings.setApi(ofSoundDevice::Api::ALSA);

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
	bpmDetector.setup(channels, sampleRate, 64);

	audioType = AudioType::TIME;
	if (audioType == AudioType::SPECTRAL) {
		int numBands = 24;
		vector<Component> melBands(numBands);
		for (int i = 0; i < numBands; i++) {
			melBands[i].value.x = float(i) / numBands;
		}
		biomass.setupAudio(melBands);
	} else if (audioType == AudioType::TIME) {
		audioArraySize = 100;
		audioArray.resize(audioArraySize);
		for (int i = 0; i < audioArraySize; i++) {
			audioArray[i].value.x = 0;
		}
		biomass.setupAudio(audioArray);
	} else if (audioType == AudioType::HEARTBEAT) {
		audioArraySize = 100;
		audioArray.resize(audioArraySize);
		for (int i = 0; i < audioArraySize; i++) {
			audioArray[i].value.x = 0;
		}
		biomass.setupAudio(audioArray);
	}

	// video and optical flow setup
	vidGrabber.setVerbose(true);
	vidGrabber.setDeviceID(0);
	int sourceWidth = mapWidth;
	int sourceHeight = mapHeight;
	vidGrabber.initGrabber(sourceWidth, sourceHeight);
	
	blurAmount = 15;
	cvDownScale = 10;
	biomass.setCVDownScale(cvDownScale);
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

	ofxUDPSettings udpSettings;
	udpSettings.receiveOn(11999);
	udpSettings.blocking = false;

	udpConnection.Setup(udpSettings);

	artificerImage.load("artitficer-title.png");
	artificerImage.resize(ofGetWidth(), ofGetHeight());
}

//--------------------------------------------------------------
void ofApp::update(){

	float bps = bpmDetector.getBPM() / 60.;
	biomass.setBPS(bps);
	// float dayRate = 4 * bps;
	float dayRate = 20;
	
	biomass.setDayRate(dayRate);

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

	int mapWidth = biomass.getMapWidth();
	int mapHeight = biomass.getMapHeight();
	int numCols = mapWidth / cvDownScale;
	int numRows = mapHeight / cvDownScale;
	
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
	vector<float> melBands = audioAnalyzer.getValues(MEL_BANDS, 0, highSmoothing);

	if (audioType == AudioType::SPECTRAL)
	{
		audioArraySize = 24;
		vector<Component> melBandsComponents(audioArraySize);
		for(int i = 0; i < audioArraySize; i++){
			melBandsComponents[i].value.x = ofMap(melBands[i], DB_MIN, DB_MAX, 0.0, 1.0, true);//clamped value
		}
		biomass.updateAudio(melBandsComponents);
	}
	else if (audioType == AudioType::TIME)
	{
		audioArraySize = 100;
		audioArray[0].value.x = ofMap(melBands[0], DB_MIN, DB_MAX, 0.0, 1.0, true);
		for (int i = audioArraySize - 1; i > 0; i--) {
			audioArray[i].value.x = audioArray[i - 1].value.x;
		}
		biomass.updateAudio(audioArray);
	}
	else if (audioType == AudioType::HEARTBEAT)
	{
		// receive network data
		char udpMessage[100000];
		udpConnection.Receive(udpMessage,100000);
		string allMessages = udpMessage;
		if (allMessages != "") {
			vector<string> messages = ofSplitString(allMessages,"\n");
			// remove empty messages
			for (int i = 0; i < messages.size(); i++) {
				if (messages[i] == "") {
					messages.erase(messages.begin() + i);
				}
			}
			if (messages.size() > 0) {
				for (int i = audioArraySize - 1; i >= messages.size(); i--) {
					audioArray[i].value.x = audioArray[i - 1].value.x;
				}
			}
			for(int i=0;i<messages.size();i++){
				vector<string> message = ofSplitString(messages[i],",");
				float bpm = atof(message[0].c_str());
				float heartbeat = atof(message[1].c_str());
				audioArray[messages.size() - i - 1].value.x = heartbeat;
			}
			if (messages.size() > 0) {
				biomass.updateAudio(audioArray);
			}
		}	
	}

	if (bReloadShader) {
		postprocess.load("generic.vert", "postprocess.frag");
		biomass.reloadShaders();
		bReloadShader = false;
	}
	
	evolution.update();
	biomass.update(optFlowTexture);
}

//--------------------------------------------------------------
void ofApp::draw() {
	float time = ofGetElapsedTimef();

	float bps = bpmDetector.getBPM() / 60.;
	vector<float> melBands = audioAnalyzer.getValues(MEL_BANDS, 0, highSmoothing);

	ofTexture tex;
	fbo.begin();
	biomass.draw();
	fbo.end();

	evolution.evaluate(fbo);

	bool postprocessEnabled = false;

	if (postprocessEnabled) {
		tex = fbo.getTexture();

		fbo.begin();
		postprocess.begin();
		postprocess.setUniformTexture("tex", tex, 0);
		postprocess.setUniformTexture("artificer", artificerImage.getTexture(), 1);
		postprocess.setUniform2i("mapSize", biomass.getMapWidth(), biomass.getMapHeight());
		postprocess.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
		postprocess.setUniform1f("time", time);
		postprocess.setUniform1f("bps", bps);
		postprocess.setUniform1f("bass", melBands[0]);
		ofSetColor(255);
		ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
		postprocess.end();
		fbo.end();
	}

	fbo.draw(0, 0);

	ofDrawBitmapString(ofGetFrameRate(),20,20);
}

void ofApp::exit(){
	biomass.exit();
	soundStream.close();

	midiIn.closePort();
	midiIn.removeListener(this);
}

//--------------------------------------------------------------
void ofApp::audioIn(ofSoundBuffer & buffer){
	if (buffer.getNumChannels() == audioAnalyzer.getChannelAnalyzersPtrs().size()) {
		buffer *= volume;
		audioAnalyzer.analyze(buffer);
		bpmDetector.processFrame(buffer.getBuffer().data(), buffer.getNumFrames(), buffer.getNumChannels());
	}
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
			float newReactionFlowMag = ofMap(val, 0, 127, 0., 30.);
			biomass.setReactionFlowMag(newReactionFlowMag);
		} else if (message.status == MIDI_PITCH_BEND) {
			int val = message.value;
			// 0 - 16383
			float newAgentFlowMag;
			if (val > 8192) {
				newAgentFlowMag = ofMap(val, 8192, 16383, 0., 30.);
			} else {
				newAgentFlowMag = 0.;
			}
			biomass.setAgentFlowMag(newAgentFlowMag);
		}
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
	if (key == '0') {
		biomass.setDisplay(0);
	} else if (key == '1') {
		biomass.setDisplay(1);
	} else if (key == '2') {
		biomass.setDisplay(2);
	} else if (key == '3') {
		biomass.setDisplay(3);
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
	
	if (key == 48) {
		biomass.reSpawnAgents();
	} else if (key == 49) {
		int display = 0;
		biomass.setDisplay(display);
	} else if (key == 50) {
		int display = 1;
		biomass.setDisplay(display);
	} else if (key == 51) {
		int display = 2;
		biomass.setDisplay(display);
	} else if (key == 52) {
		glm::vec3 newColourA = glm::vec3(1., 0., 0.);
		glm::vec3 newColourB = glm::vec3(0., 1., 0.);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 53) {
		glm::vec3 newColourA = glm::vec3(1.0, 0.906, 0.51);
		glm::vec3 newColourB = glm::vec3(0.98, 0.345, 0.118);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 54) {
		glm::vec3 newColourA = glm::vec3(0.494, 0.921, 0.063);
		glm::vec3 newColourB = glm::vec3(0.839, 0.812, 0.153);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 55) {
		glm::vec3 newColourA = glm::vec3(0.839, 0.02, 0.004);
		glm::vec3 newColourB = glm::vec3(0., 0., 1.);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 56) {
		glm::vec3 newColourA = glm::vec3(1., 0., 0.);
		glm::vec3 newColourB = glm::vec3(0., 0., 1.);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 57) {
		glm::vec3 newColourA = glm::vec3(191./255., 11./255., 59./255.);
		glm::vec3 newColourB = glm::vec3(213./255., 13./255., 216./255.);
		biomass.setReactionColour(newColourA, newColourB);
	} else if (key == 58) {
		glm::vec4 colour1 = glm::vec4(0.796, 0.2, 1., 1.);
		glm::vec4 colour2 = glm::vec4(0.1, 0.969, 1., 1.);
		biomass.setSpeciesColour(colour1, colour2);
	} else if (key == 59) {
		glm::vec4 colour1 = glm::vec4(0.263, 0.31, 0.98, 1.);
		glm::vec4 colour2 = glm::vec4(0.396, 0.839, 0.749, 1.);
		biomass.setSpeciesColour(colour1, colour2);
	} else if (key == 60) {
		float newFeedMin = 0.01;
		float newFeedRange = 0.09;
		biomass.setReactionFeedRange(newFeedMin, newFeedRange);
		biomass.reSpawnReaction();
	} else if (key == 61) {
		float newFeedMin = 0.01;
		float newFeedRange = 0.025;
		biomass.setReactionFeedRange(newFeedMin, newFeedRange);
		biomass.reSpawnReaction();
	} else if (key == 62) {
		float newFeedMin = 0.035;
		float newFeedRange = 0.025;
		biomass.setReactionFeedRange(newFeedMin, newFeedRange);
		biomass.reSpawnReaction();
	} else if (key == 63) {
		float newFeedMin = 0.06;
		float newFeedRange = 0.015;
		biomass.setReactionFeedRange(newFeedMin, newFeedRange);
		biomass.reSpawnReaction();
	} else if (key == 64) {
		float newFeedMin = 0.075;
		float newFeedRange = 0.015;
		biomass.setReactionFeedRange(newFeedMin, newFeedRange);
		biomass.reSpawnReaction();
	} else if (key == 65) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 1; i++) {
			newPoints[i].value.x = 0;
			newPoints[i].value.y = 0;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 1; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 66) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 2; i++) {
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 2;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 2; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 67) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 2; i++) {
			newPoints[i].value.x = 0.5 + 0.5 * i;
			newPoints[i].value.y = float(i) / 2;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 2; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 68) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 3; i++) {
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 3;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 69) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 3; i++) {
			newPoints[i].value.x = 1;
			newPoints[i].value.y = float(i) / 3.5;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 70) {
		vector<Component> newPoints;
		newPoints.resize(biomass.getNumPoints());
		for (int i = 0; i < 3; i++) {
			newPoints[i].value.x = 0.5 + i * 0.5;
			newPoints[i].value.y = float(i) / 3.5;
			newPoints[i].value.z = 1;
			newPoints[i].value.w = 0;
		}
		for (int i = 3; i < 5; i++) {
			newPoints[i].value = glm::vec4(0);
		}
		biomass.setPoints(newPoints);
	} else if (key == 72) {
		bReloadShader = true;
		biomass.reloadShaders();
	}
}

