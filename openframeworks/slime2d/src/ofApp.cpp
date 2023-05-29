#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	diffuseRate = 0.0001;
	decayRate = 0.98;
	trailWeight = 1;

	AgentSpawn strategy = CENTRE;

	int numParticles = 1024 * 124;
	particles.resize(numParticles);
	for(auto & p: particles){
		if (strategy == RANDOM) {
			p.pos.x = ofRandom(0, ofGetWidth());
			p.pos.y = ofRandom(0, ofGetHeight());
		} else if (strategy == CENTRE) {
			p.pos.x = ofGetWidth() / 2;
			p.pos.y = ofGetHeight() / 2;
		}
		p.angle = ofRandom(0, 2*PI);
		p.speciesIdx = 0;
	}

	int numSpecies = 1;
	allSpecies.resize(numSpecies);
	for(auto & species: allSpecies){
		species.moveSpeed = 1;
		species.sensorAngleRad = 40 * PI / 180;
		species.sensorOffsetDist = 15;
		species.turnSpeed = 0.05 * 2 * PI;
		species.colour = glm::vec4(1., 1., 1., 1.);
	}

	particlesBuffer.allocate(particles, GL_DYNAMIC_DRAW);
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	ofPixels initialTrail;
	initialTrail.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);
	ofColor initialTrailColor(0., 0., 0., 0.);
	initialTrail.setColor(initialTrailColor);

	trailMap.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA32F);
	trailMap.loadData(initialTrail);
	trailMap.bindAsImage(1, GL_READ_WRITE);

	// load shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_diffuse_decay.glsl");
	compute_decay.linkProgram();

	renderer.load("generic.vert", "shader.frag");
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();

	compute_decay.begin();
	compute_decay.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("time", ofGetElapsedTimef());
	compute_decay.setUniform1f("diffuseRate", diffuseRate);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.dispatchCompute(trailMap.getWidth()/16, trailMap.getHeight()/16, 1);
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
void ofApp::draw(){
	ofSetColor(255);
    renderer.begin();
    ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
    renderer.end();

	ofDrawBitmapString(ofGetFrameRate(),20,20);
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