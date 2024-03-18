#include "arpeggio.h"

void Arpeggio::patch() {
    addModuleOutput("signal", gain);
}

pdsp::Patchable & Arpeggio::out_signal() {
    return out("signal");
}

Arpeggio::Arpeggio(float _x, float _y, int numSpecies) {
    patch();
    setup(_x, _y, numSpecies);
}

void Arpeggio::setup(float _x, float _y, int numSpecies) {
    pos.x = _x;
    pos.y = _y;
    synths.resize(numSpecies);
    for (int i = 0; i < numSpecies; i++) {
        synths[i].setup(60.0f + (i * 12.0f));
        synths[i].out_signal() >> gain;
    }
}

Arpeggio::Arpeggio( const Arpeggio & other ) {
    patch();
    setup(other.pos.x, other.pos.y, other.synths.size());
}

void Arpeggio::setSpecies(int species, ofColor colour) {
    for (int i = 0; i < synths.size(); i++) {
        synths[i].gate.off();
    }
    synths[species].gate.trigger(1.0f);
    speciesColour = colour;
}

void Arpeggio::draw() {
    ofSetColor(speciesColour);
    ofDrawSphere(pos.x, 0, pos.y, 10);
    ofSetColor(255);
}