#include "ofApp.h"

// before looking at this check out the basics examples,
// and also the polysynth the wavesynth example

// wavetable synth that converts realtime data to a wavetable
// remember also to select the right audio output device, as ususal.
// decomment #define USE_MIDI_KEYBOARD in ofApp.h to use a midi keyboard instead of the computer keys

//--------------------------------------------------------------
void ofApp::setup(){
    
    ofSetWindowTitle("data to waveform");
    
    boidsWidth = ofGetWidth();  // try to grab at this size.
    boidsHeight = ofGetHeight();
    boidsPosX = 230;
    boidsPosY = 10;
    col = 160;			// col for getting pixels to wave
    mode = 0;

    numBins = 10;
    boids.setup( numBins );
    
    waveplot.allocate(ofGetWidth() / 2, ofGetHeight() / 8);
    
    //patching-------------------------------
    keyboard.setPolyMode( 8 );

    int voicesNum = keyboard.getVoicesNumber();

    synth.datatable.setup( numBins, numBins ); // as many samples as the webcam width
	//synth.datatable.smoothing(0.5f);

    synth.setup( voicesNum );
    for(int i=0; i<voicesNum; ++i){
        // connect each voice to a pitch and trigger output
        keyboard.out_trig(i)  >> synth.voices[i].in("trig");
        keyboard.out_pitch(i) >> synth.voices[i].in("pitch");
    }

    // patch synth to the engine
    synth.ch(0) >> engine.audio_out(0);
    synth.ch(1) >> engine.audio_out(1);

    // graphic setup---------------------------
    ofSetVerticalSync(true);
    ofDisableAntiAliasing();
    ofBackground(0);
    ofSetColor(ofColor(0,100,100));
    ofNoFill();
    ofSetLineWidth(1.0f);

    // GUI -----------------------------------
    gui.setup("", "settings.xml", 10, boidsPosY);
    gui.add( synth.ui );
    gui.add( smooth.set("wave smoothing", 0.0f, 0.0f, 0.95f) );
    smooth.addListener(this, &ofApp::smoothCall );
    smooth.set(0.3f);

    // audio / midi setup----------------------------
    engine.listDevices();
    engine.setDeviceID(0); // REMEMBER TO SET THIS AT THE RIGHT INDEX!!!!
    engine.setup( 44100, 512, 3);     
    
}

//--------------------------------------------------------------
void ofApp::update(){
    boids.update();
    
    if(true){//synth.datatable.ready() ){
		
        vector<float> ampHist = boids.getAmpHistogram();
        vector<float> alignHist = boids.getAlignHistogram();
        vector<float> peerHist = boids.getPeerHistogram();
        
        // ------------------ GENERATING THE WAVE ----------------------
        
        // a pdsp::DataTable easily convert data to a waveform in real time
        // if you don't need to generate waves in real time but
        // interpolate between already stored waves pdsp::WaveTable is a better choice
        // for example if you want to convert an image you already have to a wavetable
        
		// switch( mode ){
		// 	case 0: // converting pixels to waveform samples
        synth.datatable.begin();
        for(int n=0; n<numBins; ++n){
            float sample = ofMap(ampHist[n], 0, 1., -0.5f, 0.5f);
            synth.datatable.data(n, sample);
        }
        synth.datatable.end(false);
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
		waveplot.begin();
		ofClear(0, 0, 0, 0);
		
		ofSetColor(255);
		ofDrawRectangle(1, 1, waveplot.getWidth()-2, waveplot.getHeight()-2);
		ofTranslate(2, 2);
        ofSetColor(255);
		// switch( mode ){
		// 	case 0: // plot the raw waveforms
        ofPolyline line;
        for(int n=0; n<numBins; ++n){
            float y = ofMap(ampHist[n], 0, 1., 0., 1.);
            ofPoint p;
            p.set(waveplot.getWidth() * n / numBins, y * waveplot.getHeight());
            line.addVertex(p);
        }
        line.draw();
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
		waveplot.end();
		
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
    boids.draw(0, 0, boidsWidth, boidsHeight);
    boids.drawGui();
    
//     string info = "datatable mode (press m to change): ";
//     switch(mode){
// 		case 0: info+="raw waveform\n"; break;
// 		case 1: info+="additive synthesis\n"; break;
// 		default: break;
// 	}
	
// 	ofTranslate( 0, boidsHeight );
//     ofDrawBitmapString(info, 0, 10);

// 	ofTranslate( 0, 50 );
	waveplot.draw(ofGetWidth()/2, ofGetHeight() * 7/8);
//     ofPopMatrix();

// #ifndef USE_MIDI_KEYBOARD
//     keyboard.draw( 10, ofGetHeight() - 10, 200, 110);
// #endif

// 	ofTranslate(boidsPosX, boidsPosY);
//     ofSetColor(255);
//     ofDrawLine(  col, 0, col, boidsHeight );
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
#ifndef USE_MIDI_KEYBOARD
    // sends key messages to ofxPDSPComputerKeyboard
    keyboard.keyPressed( key ); 
#endif
    switch(key){
		case ' ':
			channel++;
			if(channel==3) channel = 0;
		break;
		case 'm': case 'M':
			mode++;
			if(mode==2) mode = 0;
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
#ifndef USE_MIDI_KEYBOARD
    // sends key messages to ofxPDSPComputerKeyboard
    keyboard.keyReleased( key );
#endif
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
}

//--------------------------------------------------------------
void ofApp::smoothCall( float & value ) {
	synth.datatable.smoothing( value  );
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
