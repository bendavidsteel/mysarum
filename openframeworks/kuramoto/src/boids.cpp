#include "boids.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void Boids::setup(int _numBins){
	compute.setupShaderFromFile(GL_COMPUTE_SHADER,"compute1.glsl");
	compute.linkProgram();
	camera.setFarClip(ofGetWidth()*10);
	numParticles = 1024 * 8;
	particles.resize(numParticles);
	for(auto & p: particles){
		p.pos.x = ofRandom(0.,ofGetWidth());
		p.pos.y = ofRandom(0.,ofGetHeight());
		p.pos.z = ofRandom(0.,ofGetHeight());
		p.pos.w = 1.;
		p.vel.x = ofRandom(-1,1);
		p.vel.y = ofRandom(-1,1);
		p.vel.z = ofRandom(-1,1);
		p.vel.w = 1.;
		p.attr.x = 0.1;//ofRandom(0.09, 0.11); // frequency of oscillation
		p.attr.y = ofRandom(0., 2 * 3.1415); // phase of oscillation
	}
	particlesBuffer.allocate(particles,GL_DYNAMIC_DRAW);
	particlesBuffer2.allocate(particles,GL_DYNAMIC_DRAW);

	vbo.setVertexBuffer(particlesBuffer,4,sizeof(Particle));
	vbo.setColorBuffer(particlesBuffer,sizeof(Particle),sizeof(glm::vec4)*3);
	vbo.enableColors();

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA32F);

	ofBackground(0);
	ofEnableBlendMode(OF_BLENDMODE_ADD);

	gui.setup();
	shaderUniforms.setName("shader params");
	shaderUniforms.add(attractionCoeff.set("attraction",0.02,0,1.));
	shaderUniforms.add(attractionMaxDist.set("attractionMaxDist",300,0,1000));
	shaderUniforms.add(alignmentCoeff.set("alignment",0.05,0,1.));
	shaderUniforms.add(alignmentMaxDist.set("alignmentMaxDist",100,0,1000));
	shaderUniforms.add(repulsionCoeff.set("repulsion",0.05,0,1.));
	shaderUniforms.add(repulsionMaxDist.set("repulsionMaxDist",25,0,1000));
	shaderUniforms.add(maxSpeed.set("maxSpeed",0.8,0.,3.));
	shaderUniforms.add(randomForce.set("randomStrength",0.1,0.,1.));
	shaderUniforms.add(fov.set("fov", 0., -1., 1.));
	shaderUniforms.add(kuramotoStrength.set("kuramotoStrength", 0., 0., 1.));
	shaderUniforms.add(kuramotoMaxDist.set("kuramotoMaxDist", 100., 0., 1000.));
	gui.add(shaderUniforms);
	gui.add(fps.set("fps",60,0,60));

	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	particlesBuffer2.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	numBins = _numBins;
	numHists = 3;
	histogramTexture.allocate(numBins, numHists, GL_R32UI);
	histogramTexture.bindAsImage(2, GL_WRITE_ONLY, GL_R32UI);

	ampHist.resize(numBins);
	alignHist.resize(numBins);
	peerHist.resize(numBins);
}

//--------------------------------------------------------------
void Boids::update(){
	float time = ofGetElapsedTimef();

	fps = ofGetFrameRate();

	compute.begin();
	compute.setUniforms(shaderUniforms);
	compute.setUniform3i("resolution", ofGetWidth(), ofGetHeight(), ofGetHeight());
	compute.setUniform1i("numAgents",particles.size());
	compute.setUniform1f("time", time);
	compute.setUniform1f("timeDelta", 1.);
	compute.setUniform1i("ampHistIdx", 1);
	compute.setUniform1i("numBins", numBins);

	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of particles be < 1024
	compute.dispatchCompute((particles.size() + 1024 -1 )/1024, 1, 1);

	compute.end();

	particlesBuffer.copyTo(particlesBuffer2);

	// convert particle data to histogram
	for(int i = 0; i < numBins; i++){
		ampHist[i] = 0;
		alignHist[i] = 0;
		peerHist[i] = 0;
	}

	Particle * ptr = (Particle *)particlesBuffer2.map(GL_READ_ONLY);

	for(int i = 0; i < particles.size(); i++){
		Particle & p = ptr[i];
		int bin = (int)(p.color.a * (numBins - 1));
		ampHist[bin] += 1/(float)particles.size();
	}

	particlesBuffer2.unmap();
}

vector<float> Boids::getAmpHistogram(){
	return ampHist;
}

vector<float> Boids::getAlignHistogram(){
	return alignHist;
}

vector<float> Boids::getPeerHistogram(){
	return peerHist;
}

//--------------------------------------------------------------
void Boids::draw(int _x, int _y, int _w, int _h){

	fbo.begin();
	ofClear(0, 0, 0, 0);

	camera.setPosition(ofGetWidth()*0.5, ofGetHeight()*0.5, -ofGetHeight());
	camera.lookAt(glm::vec3(ofGetWidth()*0.5, ofGetHeight()*0.5, ofGetHeight()*0.5));

	ofEnableBlendMode(OF_BLENDMODE_ADD);
	camera.begin();

	ofSetColor(255,70);
	glPointSize(5);
	vbo.draw(GL_POINTS,0,particles.size());
	// ofSetColor(255);
	// glPointSize(2);
	// vbo.draw(GL_POINTS,0,particles.size());

	ofNoFill();
	ofSetColor(255, 255, 255, 255);
	ofDrawBox(ofGetWidth()/2, ofGetHeight()/2, ofGetHeight()/2, ofGetWidth(), ofGetHeight(), ofGetHeight());

	camera.end();
	fbo.end();

	ofSetColor(255);
	fbo.draw(_x, _y, _w, _h);
}

void Boids::drawGui(){
	ofSetColor(255);

	ofEnableBlendMode(OF_BLENDMODE_ALPHA);
	ofSetColor(255);
	gui.draw();
}
