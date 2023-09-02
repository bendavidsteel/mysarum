#pragma once

#include "biomass.h"

class Evolution{
    public:
        void setup(Biomass& biomass);
        void evalute(Fbo output);
        void update();
};