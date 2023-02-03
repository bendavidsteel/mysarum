

int BACKGROUND_COLOUR = 0;

int NUM_AGENTS = 5000;
int PIXEL_SCALE = 2;

Moulds moulds;

void setup() {
  size(1200, 600);
  moulds = new Moulds(NUM_AGENTS, PIXEL_SCALE);
  //food = new Food()
}

void draw() {
  background(BACKGROUND_COLOUR);
  //foodMap = food.Update();
  moulds.Update();
  //food.Draw(foodMap);
}
