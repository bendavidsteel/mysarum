/**
 * Flocking 
 * by Daniel Shiffman.  
 * 
 * An implementation of Craig Reynold's Boids program to simulate
 * the flocking behavior of birds. Each boid steers itself based on 
 * rules of avoidance, alignment, and coherence.
 * 
 * Click the mouse to add a new boid.
 */

int NUM_BOIDS = 500;
int BACKGROUND_COLOUR = 0;

Flock flock;

void setup() {
  size(1000, 500);
  flock = new Flock(NUM_BOIDS);
}

void draw() {
  background(BACKGROUND_COLOUR);
  flock.Run();
}
