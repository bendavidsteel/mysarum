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
    
    //patching-------------------------------

    selfOrganisingSynths.resize(numBins);
    boidSynths.resize(numBins);

    for (int i = 0; i < numBins; i++) {
        float pitch = ofMap(i, 0, numBins - 1, 60, 84);
        selfOrganisingSynths[i].setup(pitch);
        selfOrganisingSynths[i] >> engine.audio_out(0);
        selfOrganisingSynths[i] >> engine.audio_out(1);

        pitch = ofMap(i, 0, numBins - 1, 84, 96);
        boidSynths[i].setup(pitch);
        boidSynths[i] >> engine.audio_out(0);
        boidSynths[i] >> engine.audio_out(1);
    }

    physarumSynth.datatable.setup(numBins);
    physarumSynth >> engine.audio_out(0);
    physarumSynth >> engine.audio_out(1);

    // synth.datatable.setup( numBins, numBins ); // as many samples as the webcam width
	//synth.datatable.smoothing(0.5f);

    // synth.setup( voicesNum );
    // for(int i=0; i<voicesNum; ++i){
    //     // connect each voice to a pitch and trigger output
    //     keyboard.out_trig(i)  >> synth.voices[i].in("trig");
    //     keyboard.out_pitch(i) >> synth.voices[i].in("pitch");
    // }

    // // patch synth to the engine
    // synth.ch(0) >> engine.audio_out(0);
    // synth.ch(1) >> engine.audio_out(1);

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
    windStrength += ofRandom(-0.01, 0.01);
    if (windStrength < 0.) {
        windStrength = 0.;
    } else if (windStrength > 0.1) {
        windStrength = 0.1;
    }

    float windAngle = windDirection.angleRad(ofVec2f(1, 0));
    windAngle += ofRandom(-0.1, 0.1);
    windDirection = ofVec2f(std::cos(windAngle), std::sin(windAngle));

    boids.update(windStrength, windDirection);
    physarum.update(windStrength, windDirection);
    selfOrganising.update(windStrength, windDirection);
    noise.update();
    
    if(true){//synth.datatable.ready() ){
		
        vector<float> ampHist = boids.getAmpHistogram();
        vector<float> alignHist = boids.getAlignHistogram();
        vector<float> peerHist = boids.getPeerHistogram();

        vector<float> maxSenseHist = physarum.getMaxSenseHist();
        vector<float> avgSenseHist = physarum.getAvgSenseHist();
        vector<float> turnSpeedHist = physarum.getTurnSpeedHist();

        vector<float> newMetamerHist = selfOrganising.getNewMetamerHist();

        physarumSynth.datatable.begin();
        for (int i = 0; i < numBins; i++) {
            float metamerVal = newMetamerHist[i];
            if (metamerVal > 0.) {
                selfOrganisingSynths[i].gate.trigger(metamerVal);
            } else {
                selfOrganisingSynths[i].gate.off();
            }

            float boidAmpVal = ampHist[i];
            boidSynths[i].amp.set(boidAmpVal);

            float physarumAmpVal = maxSenseHist[i];
            physarumSynth.datatable.data(i, physarumAmpVal);
        }
        physarumSynth.datatable.end(true);
        
        // ------------------ GENERATING THE WAVE ----------------------
        
        // a pdsp::DataTable easily convert data to a waveform in real time
        // if you don't need to generate waves in real time but
        // interpolate between already stored waves pdsp::WaveTable is a better choice
        // for example if you want to convert an image you already have to a wavetable
        
		// switch( mode ){
		// 	case 0: // converting pixels to waveform samples
        // synth.datatable.begin();
        // for(int n=0; n<numBins; ++n){
        //     float sample = ofMap(ampHist[n], 0, 1., -0.5f, 0.5f);
        //     synth.datatable.data(n, sample);
        // }
        // synth.datatable.end(false);
		// 	break; // remember, raw waveform could have DC offsets, we have filtered them in the synth using an hpf
			
		// 	case 1: // converting pixels to partials for additive synthesis
		// 		synth.datatable.begin();
		// 		for(int n=0; n<numBins; ++n){
		// 			float partial = ofMap(ampHist[n], 0, 1., 0.0f, 1.0f);
		// 			synth.datatable.data(n, partial);
		// 		}
		// 		synth.datatable.end(true);
		// 	break;
		// }
		
		// ----------------- PLOTTING THE WAVEFORM ---------------------
		// waveplot.begin();
		// ofClear(0, 0, 0, 0);
		
		// ofSetColor(255);
		// ofDrawRectangle(1, 1, waveplot.getWidth()-2, waveplot.getHeight()-2);
		// ofTranslate(2, 2);
        // ofSetColor(255);
		// // switch( mode ){
		// // 	case 0: // plot the raw waveforms
        // ofPolyline line;
        // for(int n=0; n<numBins; ++n){
        //     float y = ofMap(ampHist[n], 0, 1., 0., 1.);
        //     ofPoint p;
        //     p.set(waveplot.getWidth() * n / numBins, y * waveplot.getHeight());
        //     line.addVertex(p);
        // }
        // line.draw();
		// 	break;
			
		// 	case 1: // plot the partials
		// 		for(int n=0; n<numBins; ++n){
		// 			float partial = ofMap(ampHist[n], 0, 255, 0.0f, 1.0f);
		// 			int h = waveplot.getHeight() * partial;
		// 			int y = waveplot.getHeight() - h;
		// 			ofDrawLine(n*2, y, n*2, numBins );
		// 		}
		// 	break;
		// }
		// waveplot.end();
		
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
	float camY = centreY - depth * 0.2;//(camDist * (0.5 * std::sin(timeOfDay / 3) + 0.5));
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
    
    boids.drawGui(10, 10);
    physarum.drawGui(10, ofGetHeight() * 0.25 + 10);

    ofDrawBitmapString("fps: " + ofToString(ofGetFrameRate()), ofGetWidth() - 100, 20);

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
