#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){
	lowSmoothing = 0.6;
	highSmoothing = 0.8;
	numPoints = 1.;

	sampleRate = 48000;
    bufferSize = 1024;
    channels = 2;
    
    audioAnalyzer.setup(sampleRate, bufferSize, channels);

	// int numBands = 24;
	// vector<MelBand> melBands(numBands);
	// for (int i = 0; i < numBands; i++) {
	// 	melBands[i].value.x = float(i) / numBands;
	// }

	audioBufferSize = 32;
	rmsBuffer.resize(audioBufferSize);
	for (int i = 0; i < audioBufferSize; i++) {
		rmsBuffer[i].value.x = 0.;
	}

	audioBuffer.allocate(rmsBuffer, GL_DYNAMIC_DRAW);
	audioBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	audio_texture.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA16);
	audio_texture.bindAsImage(0, GL_READ_WRITE);

	flowMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);
	flowMap.bindAsImage(1, GL_READ_WRITE);

	compute_flow.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_flow.glsl");
	compute_flow.linkProgram();

	compute_audio.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_audio.glsl");
	compute_audio.linkProgram();

	ofSoundStreamSettings settings;

	// if you want to set the device id to be different than the default
	// auto devices = soundStream.getDeviceList();
	// settings.device = devices[4];

	// you can also get devices for an specific api
	// auto devices = soundStream.getDeviceList(ofSoundDevice::Api::ALSA);
	// if (!devices.empty()) {
	// 	settings.setInDevice(devices[0]);
	// }

	// or get the default device for an specific api:
	// settings.setApi(ofSoundDevice::Api::JACK);

	// or by name
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

	ofxUDPSettings udpSettings;
	udpSettings.receiveOn(11999);
	udpSettings.blocking = false;

	udpConnection.Setup(udpSettings);
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	bool normalize = true;
	float rms = audioAnalyzer.getValue(RMS, 0, highSmoothing, normalize);
	float dissonance = audioAnalyzer.getValue(DISSONANCE, 0, highSmoothing, normalize);
	float inharmonicity = audioAnalyzer.getValue(INHARMONICITY, 0, highSmoothing, normalize);
	float centroid = audioAnalyzer.getValue(CENTROID, 0, highSmoothing, normalize);

	vector<float> tristimulus = audioAnalyzer.getValues(TRISTIMULUS, 0, lowSmoothing);
	vector<float> melBands = audioAnalyzer.getValues(MEL_BANDS, 0, lowSmoothing);

	bool isOnset = audioAnalyzer.getOnsetValue(0);

	// char udpMessage[100000];
	// udpConnection.Receive(udpMessage,100000);
	// string allMessages = udpMessage;
	// if (allMessages != "") {
	// 	float x,y;
	// 	vector<string> messages = ofSplitString(allMessages,"\n");
	// 	// remove empty messages
	// 	for (int i = 0; i < messages.size(); i++) {
	// 		if (messages[i] == "") {
	// 			messages.erase(messages.begin() + i);
	// 		}
	// 	}
	// 	if (messages.size() > 0) {
	// 		for (int i = audioBufferSize - 1; i >= messages.size(); i--) {
	// 			rmsBuffer[i].value.x = rmsBuffer[i - 1].value.x;
	// 		}
	// 	}
	// 	for(int i=0;i<messages.size();i++){
	// 		vector<string> message = ofSplitString(messages[i],",");
	// 		float bpm = atof(message[0].c_str());
	// 		float heartbeat = atof(message[1].c_str());
	// 		rmsBuffer[messages.size() - i - 1].value.x = heartbeat;
	// 	}
	// }	

	// int numBands = 24;
	// vector<MelBand> melBandsComponents(numBands);
	// for(int i = 0; i < numBands; i++){
	// 	melBandsComponents[i].value.x = melBands[i];
	// }
	rms /= 4;
	rmsBuffer[0].value.x = rms;
	for (int i = audioBufferSize - 1; i > 0; i--) {
		rmsBuffer[i].value.x = rmsBuffer[i - 1].value.x;
	}

	// rms = 1;

	audioBuffer.updateData(rmsBuffer);

	int workGroupSize = 20;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	// modulate using a v simple neural net with weights from the audio analysis

	float days = time / 5;
	float time_of_day = fmod(days, float(2 * PI)) - PI;
	float sun_x = (ofGetWidth() / 2) + (2 * ofGetWidth() / 3) * cos(time_of_day);
	float sun_y = (ofGetHeight() / 2) + (2 * ofGetHeight() / 3) * sin(time_of_day);
	float sun_z = 25. + 15. * cos(days / 10);

	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_flow.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_flow.end();

	compute_audio.begin();
	compute_audio.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_audio.setUniform1f("deltaTime", deltaTime);
	compute_audio.setUniform1i("bufferSize", audioBufferSize);
	compute_audio.setUniform1f("angle", time_of_day);
	compute_audio.setUniform1f("rms", rms);
	compute_audio.setUniform1f("dissonance", dissonance);
	compute_audio.setUniform1f("numPoints", numPoints);
	compute_audio.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_audio.end();
}

//--------------------------------------------------------------
void ofApp::draw() {
	audio_texture.draw(0, 0, ofGetWidth(), ofGetHeight());

	ofDrawBitmapString(ofGetFrameRate(),20,20);
}

void ofApp::exit(){
	audioAnalyzer.exit();
}

void ofApp::audioIn(ofSoundBuffer & buffer){
	audioAnalyzer.analyze(buffer);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
	
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