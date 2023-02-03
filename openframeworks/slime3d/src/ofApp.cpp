#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	diffuseRate = 0.05;
	decayRate = 0.98;
	trailWeight = 1;

	int depth = 100;

	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	float angle = 30 * PI / 180;
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = 1.0; // turnStrength
	allSpecies[0].movementAttributes.z = CENTRE; //spawn
	allSpecies[0].sensorAttributes.x = 30; // sensorDist
	allSpecies[0].sensorAttributes.y = allSpecies[0].sensorAttributes.x * sin(angle); // sensorOffset
	allSpecies[0].sensorAttributes.z = allSpecies[0].sensorAttributes.y / tan(angle); // sensorOffDist
	allSpecies[0].colour = glm::vec4(0.796, 0.059, 1., 1.);

	angle = 60 * PI / 180;
	allSpecies[1].movementAttributes.x = 0.9;
	allSpecies[1].movementAttributes.y = 0.6;
	allSpecies[1].movementAttributes.z = RING;
	allSpecies[1].sensorAttributes.x = 40;
	allSpecies[1].sensorAttributes.y = allSpecies[1].sensorAttributes.x * sin(angle);
	allSpecies[1].sensorAttributes.z = allSpecies[1].sensorAttributes.y / tan(angle);
	allSpecies[1].colour = glm::vec4(0., 0.969, 1., 1.);
	

	int numParticles = 1024 * 512;
	particles.resize(numParticles);

	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, ofGetWidth());
			p.pos.y = ofRandom(0, ofGetHeight());
			p.pos.z = ofRandom(0, depth);
		} else if (allSpecies[speciesIdx].movementAttributes.z == CENTRE) {
			p.pos.x = ofGetWidth() / 2;
			p.pos.y = ofGetHeight() / 2;
			p.pos.z = depth / 2;
		} else if (allSpecies[speciesIdx].movementAttributes.z == RING) {
			float angle = ofRandom(0, 2*PI);
			float radius = 0.4 * ofGetWidth();
			p.pos.x = (ofGetWidth() / 2) + (radius * ofRandom(0.999, 1.001) * cos(angle));
			p.pos.y = (ofGetHeight() / 2) + (radius * ofRandom(0.999, 1.001) * sin(angle));
			p.pos.z = ofRandom(0, depth);
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

	render.load("generic.vert", "shader.frag");
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
    render.begin();
    ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
    render.end();

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