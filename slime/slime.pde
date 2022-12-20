

int NUM_AGENTS = 100;
int BACKGROUND_COLOUR = 0;

Mold mold;

void setup() {
  size(1000, 500);
  mold = new Mold(NUM_AGENTS);
}

void draw() {
  background(BACKGROUND_COLOUR);
  mold.run();
}
