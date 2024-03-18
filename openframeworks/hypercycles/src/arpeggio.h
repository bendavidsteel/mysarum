#pragma once

#include "ofMain.h"

#include "synths.h"

class Arpeggio : public pdsp::Patchable {
    public:
        Arpeggio(float _x, float _y, int numSpecies); // default constructor
        Arpeggio( const Arpeggio & other ); // you need this to use std::vector with your class, otherwise will not compile
        void setup(float _x, float _y, int numSpecies);
        void draw();
        void patch();
        pdsp::Patchable & out_signal();
        void setSpecies(int species, ofColor colour);
        ofColor speciesColour;

        ofVec2f pos;
        vector<HypercyclesSynth> synths;
        pdsp::ParameterGain gain;
};