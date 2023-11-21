#include "ofApp.h"

// before looking at this check out the basics examples,
// and also the polysynth the wavesynth example

// wavetable synth that converts realtime data to a wavetable
// remember also to select the right audio output device, as ususal.
// decomment #define USE_MIDI_KEYBOARD in ofApp.h to use a midi keyboard instead of the computer keys

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofSetWindowTitle("data to waveform");

    width = 640;
    height = 640;
    depth = 640;

    numBins = 10;
    boids.setup(numBins, width, height, depth);
    physarum.setup(numBins, width, height, depth);
    selfOrganising.setup(numBins, width, height, depth);
    noise.setup(width, depth);

    windStrength = 0.;
    windDirection = ofVec2f(0, 0);

    boidActivity = 0.;
    physarumActivity = 0.;
    selfOrganisingActivity = 0.;
    
    //patching-------------------------------

    selfOrganisingSynths.resize(numBins);
    boidSynths.resize(numBins);
    boidsAmpCtrls.resize(numBins);
    boidsPitchCtrls.resize(numBins);

    for (int i = 0; i < numBins; i++) {
        float pitch = ofMap(i, 0, numBins - 1, 60, 84);
        selfOrganisingSynths[i].setup(pitch);
        selfOrganisingSynths[i] >> engine.audio_out(0);
        selfOrganisingSynths[i] >> engine.audio_out(1);

        boidsPitchCtrls[i] >> boidSynths[i].in("pitch");
        boidsAmpCtrls[i] >> boidSynths[i].in_amp();
        boidSynths[i] * dB(-24.0f) >> engine.audio_out(0);
        boidSynths[i] * dB(-24.0f) >> engine.audio_out(1);

        pitch = ofMap(i, 0, numBins - 1, 84.0f, 96.0f);
        boidsPitchCtrls[i].set(pitch);
        boidsAmpCtrls[i].set(0.);
    }

    physarumSynth.datatable.setup(numBins, numBins);
    physarumSynth.setup(1);

    physarumTrigger.out_trig() >> physarumSynth.voices[0].in("trig");
    physarumPitch >> physarumSynth.voices[0].in("pitch");

    physarumSynth.ch(0) * dB(2.0f) >> engine.audio_out(0);
    physarumSynth.ch(1) * dB(2.0f) >> engine.audio_out(1);


    // graphic setup---------------------------
    ofSetVerticalSync(true);
    ofDisableAntiAliasing();
    ofBackground(0);
    ofSetColor(ofColor(0,100,100));
    ofNoFill();
    ofSetLineWidth(1.0f);

    cam.setFarClip(ofGetWidth()*10);
	cam.setNearClip(0.1);

    // GUI -----------------------------------
    // gui.setup("", "settings.xml", 10, 10);
    // // gui.add( synth.ui );
    // gui.add( smooth.set("wave smoothing", 0.0f, 0.0f, 0.95f) );
    // smooth.addListener(this, &ofApp::smoothCall );
    // smooth.set(0.3f);

    // audio / midi setup----------------------------
    engine.listDevices();
    engine.setDeviceID(0); // REMEMBER TO SET THIS AT THE RIGHT INDEX!!!!
    engine.setup( 44100, 512, 3);     
    
}

//--------------------------------------------------------------
void ofApp::update(){
    windStrength += ofRandom(-0.005, 0.005);
    if (windStrength < 0.) {
        windStrength = 0.;
    } else if (windStrength > 0.1) {
        windStrength = 0.1;
    }

    float time = ofGetElapsedTimef();
    if (time > 20. && selfOrganisingActivity < 1.0) {
        selfOrganisingActivity += 0.01;
    }

    if (time > 10. && physarumActivity < 1.0) {
        physarumActivity += 0.01;
    }

    if (time > 0. && boidActivity < 1.0) {
        boidActivity += 0.01;
    }

    // selfOrganisingAmp.set(selfOrganisingActivity);
    // boidAmp.set(boidActivity);
    // physarumAmp.set(physarumActivity);

    float windAngle = windDirection.angleRad(ofVec2f(1, 0));
    windAngle += ofRandom(-0.1, 0.1);
    windDirection = ofVec2f(std::cos(windAngle), std::sin(windAngle));

    boids.update(windStrength, windDirection, boidActivity);
    physarum.update(windStrength, windDirection, physarumActivity);
    selfOrganising.update(windStrength, windDirection, selfOrganisingActivity);
    noise.update();
    
    if(true){//synth.datatable.ready() ){
		
        vector<float> ampHist = boids.getAmpHistogram();
        vector<float> alignHist = boids.getAlignHistogram();
        vector<float> peerHist = boids.getPeerHistogram();

        vector<float> maxSenseHist = physarum.getMaxSenseHist();
        vector<float> avgSenseHist = physarum.getAvgSenseHist();
        vector<float> turnSpeedHist = physarum.getTurnSpeedHist();

        vector<float> newMetamerHist = selfOrganising.getNewMetamerHist();

        float maxTurnSpeed = 0.;
        float maxAvgSense = 0.;

        physarumSynth.datatable.begin();
        for (int i = 0; i < numBins; i++) {
            float metamerVal = newMetamerHist[i];
            if (metamerVal > 0. && selfOrganisingActivity > 0.) {
                selfOrganisingSynths[i].gate.trigger(metamerVal);
            } else {
                selfOrganisingSynths[i].gate.off();
            }

            float boidAmpVal = ampHist[i];
            boidsAmpCtrls[i].set(boidAmpVal * boidActivity);
            float boidsPeerVal = peerHist[i];
            float boidsPitch = ofMap(boidsPeerVal, 0., 1., 84.0f, 96.0f);
            boidsPitchCtrls[i].set(boidsPitch);

            float physarumAmpVal = maxSenseHist[i];
            physarumSynth.datatable.data(i, physarumAmpVal);

            float physarumTurnSpeedVal = turnSpeedHist[i];
            if (physarumTurnSpeedVal > maxTurnSpeed) {
                maxTurnSpeed = physarumTurnSpeedVal;
            }
            float physarumAvgSenseVal = avgSenseHist[i];
            if (physarumAvgSenseVal > maxAvgSense) {
                maxAvgSense = physarumAvgSenseVal;
            }
        }
        physarumSynth.datatable.end(true);

        physarumTrigger.trigger(maxTurnSpeed * physarumActivity);
        float pitch = ofMap(maxAvgSense, 0., 1., 40.f, 48.f);
        physarumPitch.set(pitch);
    }
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    // draw GUI
    // gui.draw();

	// ofPushMatrix();
    // // draw some keyboard keys and infos
	// switch(channel){
	// 	case 0: ofSetColor(255, 0, 0); break;
	// 	case 1: ofSetColor(0, 255, 0); break;
	// 	case 2: ofSetColor(0, 0, 255); break;
	// 	default: break;
	// }
	// ofTranslate(boidsPosX, boidsPosY);

    float centreX = width/2;
	float centreY = height/2;
	float centreZ = depth/2;
	cam.lookAt(glm::vec3(centreX, centreY, centreZ));

	float timeOfDay = ofGetElapsedTimef() * 0.1;
	float camDist = width * 1.1;
	float camX = centreX + (camDist * std::sin(timeOfDay));
	float camY = centreY - (depth * 0.2) + (depth * 0.2 * std::sin(timeOfDay / 3));
	float camZ = centreZ + (camDist * std::cos(timeOfDay));
	// float camX = centreX;
	// float camY = centreY - camDist;
	// float camZ = centreZ + camDist;
	cam.setPosition(camX, camY, camZ);

	cam.begin();

    ofEnableDepthTest();
    // ofEnableBlendMode(OF_BLENDMODE_ADD);
    
    noise.draw();
    selfOrganising.draw();
    boids.draw();
    ofEnableBlendMode(OF_BLENDMODE_SCREEN);
    physarum.draw();
    ofEnableBlendMode(OF_BLENDMODE_ALPHA);
    
    ofDisableDepthTest();

    ofSetColor(255);
    ofDrawBox(width/2, height/2, depth/2, width, height, depth);

    cam.end();
    
    bool drawGui = false;
    if (drawGui) {
        boids.drawGui(10, 10);
        physarum.drawGui(10, ofGetHeight() * 0.25 + 10);

        ofDrawBitmapString("fps: " + ofToString(ofGetFrameRate()), ofGetWidth() - 100, 20);
    }

// //     string info = "datatable mode (press m to change): ";
// //     switch(mode){
// // 		case 0: info+="raw waveform\n"; break;
// // 		case 1: info+="additive synthesis\n"; break;
// // 		default: break;
// // 	}
	
// // 	ofTranslate( 0, boidsHeight );
// //     ofDrawBitmapString(info, 0, 10);

// // 	ofTranslate( 0, 50 );
// 	waveplot.draw(ofGetWidth()/2, ofGetHeight() * 7/8);
//     ofPopMatrix();


}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){}
//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){}
//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){}
//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){}
//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){}
//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){}
//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){}
