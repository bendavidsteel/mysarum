#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){
	smoothing = 0.6;

	sampleRate = 44100;
    bufferSize = 512;
    channels = 2;
    
    audioAnalyzer.setup(sampleRate, bufferSize, channels);

	int numBands = 24;
	vector<MelBand> melBands(numBands);
	for (int i = 0; i < numBands; i++) {
		melBands[i].value.x = float(i) / numBands;
	}
	melBandsBuffer.allocate(melBands, GL_DYNAMIC_DRAW);
	melBandsBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	audio_texture.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA16);
	audio_texture.bindAsImage(0, GL_READ_WRITE);

	compute_audio.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_audio.glsl");
	compute_audio.linkProgram();

	ofSoundStreamSettings settings;

	// if you want to set the device id to be different than the default
	// auto devices = soundStream.getDeviceList();
	// settings.device = devices[4];

	// you can also get devices for an specific api
	// auto devices = soundStream.getDevicesByApi(ofSoundDevice::Api::PULSE);
	// settings.device = devices[0];

	// or get the default device for an specific api:
	// settings.api = ofSoundDevice::Api::PULSE;

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

}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	bool normalize = true;
	float dissonance = audioAnalyzer.getValue(DISSONANCE, 0, smoothing, normalize);
	float inharmonicity = audioAnalyzer.getValue(INHARMONICITY, 0, smoothing, normalize);
	float centroid = audioAnalyzer.getValue(CENTROID, 0, smoothing, normalize);

	vector<float> tristimulus = audioAnalyzer.getValues(TRISTIMULUS, 0, smoothing);
	vector<float> melBands = audioAnalyzer.getValues(MEL_BANDS, 0, smoothing);

	int numBands = 24;
	vector<MelBand> melBandsComponents(numBands);
	for(int i = 0; i < numBands; i++){
		melBandsComponents[i].value.x = melBands[i];
	}
	melBandsBuffer.updateData(melBandsComponents);

	int workGroupSize = 32;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

	// modulate using a v simple neural net with weights from the audio analysis

	float days = time / 5;
	float time_of_day = fmod(days, float(2 * PI)) - PI;
	float sun_x = (ofGetWidth() / 2) + (2 * ofGetWidth() / 3) * cos(time_of_day);
	float sun_y = (ofGetHeight() / 2) + (2 * ofGetHeight() / 3) * sin(time_of_day);
	float sun_z = 25. + 15. * cos(days / 10);

	compute_audio.begin();
	compute_audio.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_audio.setUniform1f("deltaTime", deltaTime);
	compute_audio.setUniform1i("numBands", numBands);
	compute_audio.setUniform1f("angle", time_of_day);
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