#include "ofApp.h"
#include "ofConstants.h"

//--------------------------------------------------------------
void ofApp::setup(){

	// constants

	// slime constants
	volWidth = ofGetWidth();
	volHeight = ofGetHeight();
	volDepth = 1;

	diffuseRate = 0.2;
	decayRate = 0.98;
	trailWeight = 1;

	// general constants
	sampleRate = 44100;
    channels = 2;

	fileName = "testMovie";
    fileExt = ".mov"; // ffmpeg uses the extension to determine the container type. run 'ffmpeg -formats' to see supported formats

	// slime setup
	int numSpecies = 2;
	allSpecies.resize(numSpecies);
	float angle = 30 * PI / 180;
	allSpecies[0].movementAttributes.x = 1.1; // moveSpeed
	allSpecies[0].movementAttributes.y = 1.0; // turnStrength
	allSpecies[0].movementAttributes.z = CIRCLE; //spawn
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

	int numParticles = 1024 * 256;
	particles.resize(numParticles);

	int speciesIdx = 0;
	for(int idx = 0; idx < particles.size(); idx++){
		auto &p = particles[idx];
		speciesIdx = idx % numSpecies;
		if (allSpecies[speciesIdx].movementAttributes.z == RANDOM) {
			p.pos.x = ofRandom(0, volWidth);
			p.pos.y = ofRandom(0, volHeight);
			p.pos.z = ofRandom(0, volDepth);
		} else if (allSpecies[speciesIdx].movementAttributes.z == CIRCLE) {
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
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 4);

	allSpeciesBuffer.allocate(allSpecies, GL_DYNAMIC_DRAW);
	allSpeciesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 5);

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
	trailMap.bindAsImage(6, GL_READ_WRITE);

	// reaction diffusion setup

	ofPixels initialReaction;
	initialReaction.allocate(ofGetWidth(), ofGetHeight(), OF_PIXELS_RGBA);
	ofColor baseReactionColour(255., 0., 0., 0.);
	initialReaction.setColor(baseReactionColour);

	Spawn initialPattern = RANDOM;

	ofColor reactantColour(255., 255., 0., 0.);
	if (initialPattern == CIRCLE) {
		int circleSize = ofGetHeight() / 32;
		glm::vec2 centre = glm::vec2(ofGetWidth() / 2, ofGetHeight() / 2);
		for (int j = 0; j < ofGetHeight(); j++) {
			for (int i = 0; i < ofGetWidth(); i++) {
				glm::vec2 point = glm::vec2(i, j);
				float radius = glm::distance(point, centre);
				if (radius < circleSize) {
					initialReaction.setColor(i, j, reactantColour);
				}
			}
		}
	} else if (initialPattern == RANDOM) {
		for (int j = 0; j < ofGetHeight(); j++) {
			for (int i = 0; i < ofGetWidth(); i++) {
				if (ofRandom(1) < 0.1) {
					initialReaction.setColor(i, j, reactantColour);
				}
			}
		}
	}

	reactionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	reactionMap.loadData(initialReaction);
	reactionMap.bindAsImage(3, GL_READ_WRITE);

	feedkillMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	feedkillMap.bindAsImage(2, GL_READ_WRITE);

	diffusionMap.allocate(ofGetWidth(), ofGetHeight(), GL_RG16);
	diffusionMap.bindAsImage(1, GL_READ_WRITE);

	// general setup

	flowMap.allocate(volWidth, volHeight, volDepth, GL_RGBA8);
	flowMap.bindAsImage(0, GL_READ_WRITE);

	fbo.allocate(ofGetWidth(), ofGetHeight(), GL_RGBA8);

	// load slime shaders
	compute_agents.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_agents.glsl");
	compute_agents.linkProgram();

	compute_decay.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_decay.glsl");
	compute_decay.linkProgram();

	compute_diffuse.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_diffuse.glsl");
	compute_diffuse.linkProgram();

	// load reaction diffusion shaders

	compute_diffusion.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_diffusion.glsl");
	compute_diffusion.linkProgram();

	compute_feedkill.setupShaderFromFile(GL_COMPUTE_SHADER, "compute_feedkill.glsl");
	compute_feedkill.linkProgram();

	compute_reaction.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_reaction.glsl");
	compute_reaction.linkProgram();

	// load general shaders

	compute_flow.setupShaderFromFile(GL_COMPUTE_SHADER,"compute_flow.glsl");
	compute_flow.linkProgram();

	renderer.load("renderer.vert", "renderer.frag");

	// video recording
	ofAddListener(vidRecorder.outputFileCompleteEvent, this, &ofApp::recordingComplete);

	bRecording = false;

	// sound
	ofSoundStreamSetup(0, 2, 44100, 256, 4);

	// video and optical flow setup
	vidGrabber.setVerbose(true);
	int sourceWidth = ofGetWidth();
	int sourceHeight = ofGetHeight();
	vidGrabber.setup(sourceWidth, sourceHeight);
	
	blurAmount = 11;
	cvDownScale = 8;
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
	opticalFlowTexture.allocate(scaledWidth, scaledHeight, GL_RGBA8);
	opticalFlowTexture.bindAsImage(7, GL_READ_WRITE);
}

//--------------------------------------------------------------
void ofApp::update(){
	double deltaTime = 1.; //ofGetLastFrameTime();
	float time = ofGetElapsedTimef();

	int workGroupSize = 20;

	int widthWorkGroups = ceil(ofGetWidth()/workGroupSize);
	int heightWorkGroups = ceil(ofGetHeight()/workGroupSize);

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
	opticalFlowTexture.loadData(opticalFlowPixels);

	// general updates
	compute_flow.begin();
	compute_flow.setUniform1f("time", time);
	compute_flow.setUniform2i("resolution", volWidth, volHeight);
	compute_flow.setUniform1i("opticalFlowDownScale", cvDownScale);
	compute_flow.dispatchCompute(widthWorkGroups, heightWorkGroups, volDepth/1);
	compute_flow.end();

	// slime updates
	// horizontal blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 1, 0, 0);
	compute_diffuse.dispatchCompute(widthWorkGroups, heightWorkGroups, volDepth/1);
	compute_diffuse.end();

	// vertical blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 1, 0);
	compute_diffuse.dispatchCompute(widthWorkGroups, heightWorkGroups, volDepth/1);
	compute_diffuse.end();

	// depth blur
	compute_diffuse.begin();
	compute_diffuse.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_diffuse.setUniform1f("deltaTime", deltaTime);
	compute_diffuse.setUniform1f("diffuseRate", diffuseRate);
	compute_diffuse.setUniform3i("blurDir", 0, 0, 1);
	compute_diffuse.dispatchCompute(widthWorkGroups, heightWorkGroups, volDepth/1);
	compute_diffuse.end();

	compute_decay.begin();
	compute_decay.setUniform3i("resolution", volWidth, volHeight, volDepth);
	compute_decay.setUniform1f("deltaTime", deltaTime);
	compute_decay.setUniform1f("decayRate", decayRate);
	compute_decay.dispatchCompute(widthWorkGroups, heightWorkGroups, volDepth/1);
	compute_decay.end();

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

	// reaction diffusion updates
	compute_diffusion.begin();
	compute_diffusion.setUniform1f("time", time);
	compute_diffusion.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_diffusion.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_diffusion.end();

	compute_feedkill.begin();
	compute_feedkill.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_feedkill.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_feedkill.end();

	compute_reaction.begin();
	compute_reaction.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	compute_reaction.setUniform1f("deltaTime", deltaTime);
	compute_reaction.dispatchCompute(widthWorkGroups, heightWorkGroups, 1);
	compute_reaction.end();
}

//--------------------------------------------------------------
void ofApp::draw() {

	// opticalFlowTexture.draw(0, 0, ofGetWidth(), ofGetHeight());
	float time = ofGetElapsedTimef();
	float days = time / 30;
	float sun_x = (ofGetWidth() / 2) + (2 * ofGetWidth() / 3) * cos(days);
	float sun_y = (ofGetHeight() / 2) + (2 * ofGetHeight() / 3) * sin(days);
	float sun_z = 25. + 15. * cos(days / 10);

	fbo.begin();
	ofClear(255,255,255, 0);
	renderer.begin();
	renderer.setUniform2i("resolution", ofGetWidth(), ofGetHeight());
	renderer.setUniform1i("trail_depth", volDepth);
	renderer.setUniform3f("colourA", 1., 0., 0.);
	renderer.setUniform3f("colourB", 0., 1., 0.);
	renderer.setUniform3f("light", sun_x, sun_y, sun_z);
	renderer.setUniform1f("chem_height", 1.);
	renderer.setUniform1f("trail_height", 2.);
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
	reactionMap.clear();
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