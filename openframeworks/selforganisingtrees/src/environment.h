#include "selforganising.h"

class Environment {
    public:
        virtual void updateBudEnvironment() = 0;
        virtual void updateBudEnvironment(shared_ptr<Metamer> metamer, Tree tree) = 0;
        virtual void addToEnvironment(shared_ptr<Metamer> metamer) = 0;
        virtual void removeFromEnvironment(shared_ptr<Metamer> metamer) = 0;
}