

int BACKGROUND_COLOUR = 0;

int NUM_AGENTS = 10000;
int PIXEL_SCALE = 2;

Moulds moulds;

void setup() {
  size(1200, 600);
  moulds = new Moulds(NUM_AGENTS, PIXEL_SCALE);
}

void draw() {
  background(BACKGROUND_COLOUR);
  moulds.Update();
}
