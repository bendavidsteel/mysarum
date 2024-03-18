// ofApp.cpp
#include "ofApp.h"

void ofApp::setup(){
    ofSetFrameRate(60);
    
    // GUI setup
    gui.setup();
    gui.add(frequency.setup("Frequency", 440, 20, 2000));
    gui.add(playButton.setup("Play"));
    playButton.addListener(this, &ofApp::playButtonPressed);
    
    // Audio setup
    ofSoundStreamSettings settings;
    settings.numOutputChannels = 2;
    settings.sampleRate = 44100;
    settings.bufferSize = 512;
    settings.numBuffers = 4;
    ofSoundStreamSetup(settings);
    
    isPlaying = false;
}

void ofApp::update(){
    // Update any parameters here
}

void ofApp::draw(){
    ofBackground(0);
    
    // Draw the GUI
    gui.draw();
}


void ofApp::audioOut(float *output, int bufferSize, int nChannels){
    for (int i = 0; i < bufferSize; i++) {
        if (isPlaying) {
            double wave = osc.sinewave(frequency);
            output[i * nChannels] = wave * 0.5;
            output[i * nChannels + 1] = wave * 0.5;
        } else {
            output[i * nChannels] = 0.0;
            output[i * nChannels + 1] = 0.0;
        }
    }
}

void ofApp::playButtonPressed() {
    isPlaying = !isPlaying;
    if (isPlaying) {
        // You can add any setup you need when starting sound
    } else {
        // You can add any setup you need when stopping sound
    }
}
