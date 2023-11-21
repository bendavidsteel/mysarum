#pragma once

#include "ofMain.h"
#include "ofxPDSP.h"

class SelfOrganisingSynth : public pdsp::Patchable {
    public:
        SelfOrganisingSynth() { patch(); } // default constructor
        SelfOrganisingSynth( const SelfOrganisingSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
        
        void patch() {
            addModuleOutput("signal", gain);

            gate.out_trig() >> env.in_trig();
            env >> amp.in_mod();

            pitchCtrl >> osc.in_pitch();
            osc.out_sine() >> amp * dB(-12.0f) >> gain;
            amp * dB(-12.0f) >> gain;

            0.0f >> env.in_attack();
            50.0f >> env.in_decay();
            0.5f >> env.in_sustain();
            500.0f >> env.in_release();
        }

        void setup(float pitch) {
            pitchCtrl.set(pitch);
        }

        pdsp::Patchable & out_signal() {
            return out("signal");
        }

        pdsp::VAOscillator osc;
        pdsp::ADSR env;
        pdsp::TriggerControl gate;
        pdsp::ValueControl pitchCtrl;
        pdsp::Amp amp;
        pdsp::ParameterGain gain;
};

class BoidsSynth : public pdsp::Patchable {
    public:
        BoidsSynth() { patch(); } // default constructor
        BoidsSynth( const BoidsSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
        
        void patch() {
            addModuleOutput("signal", gain);

            pitchCtrl >> osc.in_pitch();
            osc.out_sine() >> amp * dB(-24.0f) >> gain;
            amp * dB(-24.0f) >> gain;
        }

        void setup(float pitch) {
            pitchCtrl.set(pitch);
        }

        pdsp::Patchable & out_signal() {
            return out("signal");
        }

        pdsp::VAOscillator osc;
        pdsp::ValueControl pitchCtrl;
        pdsp::Amp amp;
        pdsp::ParameterGain gain;
};

class PhysarumSynth : public pdsp::Patchable {
    public:
        PhysarumSynth() { patch(); } // default constructor
        PhysarumSynth( const PhysarumSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
        
        void patch() {
            addModuleOutput("signal", gain);

            osc.setTable(datatable);

            osc >> amp * dB(-12.0f) >> gain;
            amp * dB(-12.0f) >> gain;
        }

        pdsp::Patchable & out_signal() {
            return out("signal");
        }

        pdsp::DataOscillator osc;
        pdsp::DataTable datatable;
        pdsp::Amp amp;
        pdsp::ParameterGain gain;
};