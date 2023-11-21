#include "physarum3d.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void Physarum::setup(int _numBins, int _width, int _height, int _depth){

	numBins = _numBins;

	float volFactor = 1.;
	float heightFactor = 0.1;
	volWidth = int(_width * volFactor);
	volHeight = int(_height * volFactor * heightFactor);
	volDepth = int(_depth * volFactor);

	width = _width;
	height = int(_height * heightFactor);
	depth = _depth;

	diffuseRate.set("diffuseRate", 0.05, 0.0, 1.0);
	decayRate.set("decayRate", 0.98, 0., 1.0);
	trailWeight.set("trailWeight", 1., 0., 1.);

	trailUniforms.setName("trails");
	trailUniforms.add(diffuseRate);
	trailUniforms.add(decayRate);
	trailUniforms.add(trailWeight);

	turnSpeed.set("turnSpeed", 1.0, 0.0, 1.0);
	sensorAngleRad.set("sensorAngleRad", 15. * PI / 180., 0., 90. * PI / 180.);
	sensorOffsetDist.set("sensorOffsetDist", 5., 0., 100.);
	moveSpeed.set("moveSpeed", 1.0, 0.0, 2.0);

	speciesUniforms.setName("species");
	speciesUniforms.add(turnSpeed);
	speciesUniforms.add(sensorAngleRad);
	speciesUniforms.add(sensorOffsetDist);
	speciesUniforms.add(moveSpeed);

	int numSpecies = 1;
	allSpecies.resize(numSpecies);
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = turnSpeed; // turnSpeed
	allSpecies[0].movementAttributes.z = EDGES; //spawn
	allSpecies[0].sensorAttributes.x = sensorAngleRad * PI / 180; // sensorAngleRad
	allSpecies[0].sensorAttributes.y = sensorOffsetDist; // sensorOffsetDist
	allSpecies[0].colour = glm::vec4(1., 1., 1., 1.);

	// allSpecies[1].movementAttributes.x = 0.9;
	// allSpecies[1].movementAttributes.y = 0.6;
	// allSpecies[1].movementAttributes.z = RING;
	// allSpecies[1].sensorAttributes.x = 60 * PI / 180;
	// allSpecies[1].sensorAttributes.y = 40;
	// allSpecies[1].colour = glm::vec4(0.1, 0.969, 1., 1.);

	int numParticles = 1024 * 8;
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
		} else if (allSpecies[speciesIdx].movementAttributes.z == EDGES) {
			p.pos.y = volHeight;
			float edges = int(ofRandom(0, 4));
			if (edges == 0) {
				p.pos.x = ofRandom(0, volWidth);
				p.pos.z = 0.;
			} else if (edges == 1) {
				p.pos.x = volWidth;
				p.pos.z = ofRandom(0, volDepth);
			} else if (edges == 2) {
				p.pos.x = ofRandom(0, volWidth);
				p.pos.z = volDepth;
			} else if (edges == 3) {
				p.pos.x = 0.;
				p.pos.z = ofRandom(0, volDepth);
			}
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

	particlesBuffer2.allocate(particles, GL_DYNAMIC_DRAW);
	particlesBuffer2.bindBase(GL_SHADER_STORAGE_BUFFER, 1);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

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

	trailMap2.allocate(volWidth, volHeight, volDepth, GL_RGBA8);
	trailMap2.loadData(volumeData, volWidth, volHeight, volDepth, 0, 0, 0, GL_RGBA);

	trailMap2.bindAsImage(4, GL_READ_ONLY);
	trailMap.bindAsImage(3, GL_WRITE_ONLY);

	flowMap.allocate(volWidth, volHeight, volDepth, GL_RGBA8);
	flowMap.bindAsImage(5, GL_READ_WRITE);

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

	xyQuality.set("xyQuality", 2.0, 0.0, 2.0);
	zQuality.set("zQuality", 0.1, 0.0, 2.0);
	density.set("density", 0.2, 0.0, 1.0);
	threshold.set("threshold", 0.05, 0.0, 1.0);

	renderUniforms.setName("render");
	renderUniforms.add(xyQuality);
	renderUniforms.add(zQuality);
	renderUniforms.add(density);
	renderUniforms.add(threshold);
	ofAddListener(renderUniforms.parameterChangedE(), this, &Physarum::renderUniformsChanged);

	volume.setup(volWidth, volHeight, volDepth, ofVec3f(1,1,1),false);
	volume.setRenderSettings(xyQuality, zQuality, density, threshold);

	gui.setup();
	gui.add(trailUniforms);
	gui.add(speciesUniforms);
	gui.add(renderUniforms);

	avgSenseHist.resize(numBins);
	maxSenseHist.resize(numBins);
	turnSpeedHist.resize(numBins);
}

//--------------------------------------------------------------
void Physarum::update(float windStrength, ofVec2f windDirection){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	int localSizeX = 8;
	int localSizeY = 8;
	int localSizeZ = 8;

	compute_agents.begin();
	compute_agents.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_agents.setUniform1f("deltaTime", deltaTime);
	compute_agents.setUniform1f("time", time);
	compute_agents.setUniform1f("trailWeight", trailWeight);
	compute_agents.setUniform1f("windStrength", windStrength);
	compute_agents.setUniform2f("windDirection", windDirection.x, windDirection.y);
	compute_agents.setUniforms(speciesUniforms);
	
	// since each work group has a local_size of 1024 (this is defined in the shader)
	// we only have to issue 1 / 1024 workgroups to cover the full workload.
	// note how we add 1024 and subtract one, this is a fast way to do the equivalent
	// of std::ceil() in the float domain, i.e. to round up, so that we're also issueing
	// a work group should the total size of particles be < 1024
	compute_agents.dispatchCompute((particles.size() + 1024 -1 )/1024, 1, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	compute_agents.end();

	swapTrailMaps();

	// horizontal blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 1, 0, 0);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	compute_diffuse.end();

	swapTrailMaps();

	// vertical blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 1, 0);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	compute_diffuse.end();

	swapTrailMaps();

	// depth blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 0, 1);
	compute_diffuse.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	compute_diffuse.end();

	swapTrailMaps();

	compute_decay.begin();
	compute_decay.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	compute_decay.end();

	swapTrailMaps();

	// compute_flow.begin();
	// compute_flow.setUniform1f("time", time);
	// compute_flow.setUniform2i("resolution", volWidth, volHeight);
	// compute_flow.dispatchCompute(volWidth/localSizeX, volHeight/localSizeY, volDepth/localSizeZ);
	// compute_flow.end();

	for (int i = 0; i < numBins; i++) {
		avgSenseHist[i] = 0;
		maxSenseHist[i] = 0;
		turnSpeedHist[i] = 0;
	}

	particlesBuffer.copyTo(particlesBuffer2);

	// read particles to memory
	Agent * ptr = (Agent *) particlesBuffer.map(GL_READ_ONLY);

	for (int idx = 0; idx < particles.size(); idx++) {
		int bin = (int) ptr[idx].state.x * (numBins - 1);
		float weight = 1./ (float) particles.size();
		maxSenseHist[bin] += weight;

		bin = (int) ptr[idx].state.y * (numBins - 1);
		avgSenseHist[bin] += weight;

		bin = (int) ptr[idx].state.z * (numBins - 1);
		turnSpeedHist[bin] += weight;
	}

	particlesBuffer.unmap();
}

//--------------------------------------------------------------
void Physarum::draw() {

	volume.drawVolume(width/2, -height/2, depth/2, width, height, depth, 0);

	// fbo.begin();
	// renderer.begin();
	// renderer.setUniform2i("screen_res", ofGetWidth(), ofGetHeight());
	// renderer.setUniform3i("trail_res", volWidth, volHeight, volDepth);
	// ofSetColor(255);
	// ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	// renderer.end();
	// fbo.end();
	// fbo.draw(0, 0);
}

void Physarum::drawGui(int x, int y){
	gui.setPosition(x, y);
	// ofEnableBlendMode(OF_BLENDMODE_ALPHA);
	ofSetColor(255);
	gui.draw();
}

void Physarum::exit(){
	trailMap.clear();
	trailMap2.clear();
}

void Physarum::swapTrailMaps() {
	if (useFirstTexture) {
		trailMap.bindAsImage(4, GL_READ_ONLY);
		trailMap2.bindAsImage(3, GL_WRITE_ONLY);
	} else {
		trailMap2.bindAsImage(4, GL_READ_ONLY);
		trailMap.bindAsImage(3, GL_WRITE_ONLY);
	}

	useFirstTexture = !useFirstTexture;
}

void Physarum::renderUniformsChanged(ofAbstractParameter &e){
	volume.setRenderSettings(xyQuality, zQuality, density, threshold);
}

vector<float> Physarum::getAvgSenseHist() {
	return avgSenseHist;
}

vector<float> Physarum::getMaxSenseHist() {
	return maxSenseHist;
}

vector<float> Physarum::getTurnSpeedHist() {
	return turnSpeedHist;
}