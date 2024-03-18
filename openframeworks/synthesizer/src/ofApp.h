#pragma once
#include "ofMain.h"
#include "ofxMaxim.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp{
public:
    void setup();
    void update();
    void draw();
    void audioOut(float *output, int bufferSize, int nChannels);

    ofxPanel gui;
    ofxFloatSlider frequency;
    ofxButton playButton;
    bool isPlaying;
    
    maxiOsc osc; // Create an oscillator
    maxiMix mixer; // Create a mixer
};
