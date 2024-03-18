#pragma once

#include "ofMain.h"
#include "ofxPDSP.h"

class HypercyclesSynth : public pdsp::Patchable {
    public:
        HypercyclesSynth() { patch(); } // default constructor
        HypercyclesSynth( const HypercyclesSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
        
        void patch() {
            addModuleOutput("signal", gain);

            gate.out_trig() >> env.in_trig();
            env >> amp.in_mod();

            pitchCtrl >> osc.in_pitch();
            osc.out_triangle() >> amp * dB(-12.0f) >> gain;
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


class BoidsSynth : public pdsp::Patchable{
    
public:
    
    BoidsSynth() { patch(); } // default constructor
    BoidsSynth( const BoidsSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
    // remember that is a bad thing to copy construct in pdsp, 
    //      always just resize the vector and let the default constructor do the work
    //          resizing the vector will also disconnect everything, so do it just once before patching


    void patch (){
       
        //create inputs/outputs to be used with the in("tag") and out("tag") methods
        addModuleInput("pitch", osc.in_pitch()); // the first input added is the default input
        addModuleInput("amp", y_ctrl);
        addModuleOutput("signal", amp ); // the first output added is the default output
        
        
        // pdsp::PatchNode is a class that simply route its inputs to its output
        y_ctrl.enableBoundaries(0.0f, 1.0f); // you can clip the input of pdsp::PatchNode
        y_ctrl.set(0.0f); // and you can set its input default value
        
        //patching
        osc.out_saw() * 2.0f >> drive >> filter >> amp;
                                         y_ctrl >> amp.in_mod();        
                                         y_ctrl * 60.0f >> filter.in_cutoff();
                                                  48.0f >> filter.in_cutoff();
                                                  0.3f  >> filter.in_reso();
    }
    
    // those are optional
    pdsp::Patchable & in_pitch() {
        return in("pitch");
    }
    
    pdsp::Patchable & in_amp() {
        return in("amp");
    }
    
    pdsp::Patchable & out_signal() {
        return out("signal");
    }
    
private:

    pdsp::PatchNode     y_ctrl;
    pdsp::PatchNode     pitch_ctrl;
    pdsp::Amp           amp;
    pdsp::VAOscillator  osc;
    pdsp::Saturator1    drive; // distort the signal
    pdsp::VAFilter      filter; // 24dB multimode filter

};


// class PhysarumSynth : public pdsp::Patchable {
//     public:
//         PhysarumSynth() { patch(); } // default constructor
//         PhysarumSynth( const PhysarumSynth & other ) { patch(); } // you need this to use std::vector with your class, otherwise will not compile
        
//         void patch() {
//             addModuleOutput("signal", gain);

//             osc.setTable(datatable);

//             pitchCtrl >> osc.in_pitch();
//             osc >> amp * dB(-24.0f) >> gain;
//             amp * dB(-24.0f) >> gain;
//         }

//         void setup(float pitch) {
//             pitchCtrl.set(pitch);
//         }

//         pdsp::Patchable & out_signal() {
//             return out("signal");
//         }

//         pdsp::DataOscillator osc;
//         pdsp::DataTable datatable;
//         pdsp::ValueControl pitchCtrl;
//         pdsp::Amp amp;
//         pdsp::ParameterGain gain;
// };

// datatable based polysynth

class PhysarumSynth {

public:
    // class to rapresent each synth voice ------------
    class Voice : public pdsp::Patchable {
        friend class PhysarumSynth;
    
    public:
        Voice(){}
        Voice(const Voice& other){}
        
        float meter_mod_env() const;
        float meter_pitch() const;

    private:
        void setup(PhysarumSynth & m, int v);

        pdsp::PatchNode     voiceTrigger;
        
        pdsp::DataOscillator    oscillator;
        pdsp::VAFilter          filter;
        pdsp::Amp               amp;


        pdsp::ADSR          envelope;    
    }; // end voice class -----------------------------


    // synth public API --------------------------------------

    void setup( int numVoice );
    
    pdsp::DataTable  datatable;

    pdsp::Patchable& ch( int index );

    vector<Voice>       voices;
    ofParameterGroup    ui;

private: // --------------------------------------------------

    pdsp::ParameterGain gain;

    pdsp::Parameter     cutoff_ctrl;
    pdsp::Parameter     reso_ctrl;
    pdsp::Parameter     filter_mode_ctrl;

    pdsp::Parameter     env_attack_ctrl;
    pdsp::Parameter     env_decay_ctrl;
    pdsp::Parameter     env_sustain_ctrl;
    pdsp::Parameter     env_release_ctrl;
    pdsp::ParameterAmp  env_filter_amt;

    pdsp::Parameter     lfo_speed_ctrl;    
    pdsp::Parameter     lfo_wave_ctrl;

    pdsp::LFO           lfo;
    pdsp::Switch        lfo_switch;
    pdsp::ParameterAmp  lfo_filter_amt;    
    
    pdsp::LowCut			leakDC;  
    
    // chorus ------------------------
    pdsp::DimensionChorus   chorus;       
    ofParameterGroup    ui_chorus;
    pdsp::Parameter     chorus_speed_ctrl;
    pdsp::Parameter     chorus_depth_ctrl;
    
    std::vector<float> partials_vector;

};

