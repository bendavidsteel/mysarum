#pragma once

#include "biomass.h"

class Evolution{
    public:
        void setup(Biomass& biomass);
        void evaluate(ofFbo output);
        void update();

    private:
        Biomass biomass;
};