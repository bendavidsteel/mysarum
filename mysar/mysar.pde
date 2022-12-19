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

int NUM_BOIDS = 100;
int BACKGROUND_COLOUR = 0;

Flock flock;

void setup() {
  size(1000, 500); //<>//
  flock = new Flock();
  // Add an initial set of boids into the system
  for (int i = 0; i < NUM_BOIDS; i++) {
    flock.addBoid(new Boid(width/2,height/2));
  }
}

void draw() {
  background(BACKGROUND_COLOUR);
  noFill();
  circle(width/2, height/2, height);
  flock.run();
}

// Add a new boid into the System
void mousePressed() {
  flock.addBoid(new Boid(mouseX,mouseY));
}
