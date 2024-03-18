#include "ofApp.h"

#include "hypercycles.h"

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

    hypercycles.setup(width, depth);

    hypercycles.out_signal() >> engine.audio_out(0);
    hypercycles.out_signal() >> engine.audio_out(1);
    
    //patching-------------------------------

    // graphic setup---------------------------
    ofSetVerticalSync(true);
    ofDisableAntiAliasing();
    // ofBackground(0);
    // ofSetColor(ofColor(0,100,100));
    // ofNoFill();
    // ofSetLineWidth(1.0f);

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
    
    hypercycles.update();
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
	float centreY = 0;
	float centreZ = depth/2;
	cam.lookAt(glm::vec3(centreX, centreY, centreZ));

	float timeOfDay = 0.;//ofGetElapsedTimef() * 0.1;
	float camDist = width * 0.6;
	float camX = centreX + (camDist * std::sin(timeOfDay));
	float camY = centreY + (depth * 0.4);// + (depth * 0.2 * std::sin(timeOfDay / 3));
	float camZ = centreZ + (camDist * std::cos(timeOfDay));
	// float camX = centreX;
	// float camY = centreY - camDist;
	// float camZ = centreZ + camDist;
	cam.setPosition(camX, camY, camZ);

	cam.begin();

    // ofEnableDepthTest();
    // ofEnableBlendMode(OF_BLENDMODE_ADD);
    
    hypercycles.draw();
    
    // ofDisableDepthTest();

    // ofSetColor(255);
    // ofDrawBox(width/2, height/2, depth/2, width, height, depth);

    cam.end();
    
    bool drawGui = true;
    if (drawGui) {
        hypercycles.drawGui();
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
