// The Boid class

int BOID_COLOUR = 255;
float BOID_SIZE = 2.0;
int MAX_SPEED = 2; // Maximum speed
float MAX_FORCE = 0.03; // Maximum steering force

float WAN_FACTOR = 0.05;
float SEP_FACTOR = 2.0;
float ALI_FACTOR = 1.0;
float COH_FACTOR = 1.0;
float BOR_FACTOR = 1.0;

float EPSILON = 0.00001;

class Boid {

  PVector position;
  PVector velocity;
  PVector acceleration; 

  Boid(float x, float y) {
    acceleration = new PVector(0, 0);

    // This is a new PVector method not yet implemented in JS
    velocity = PVector.random2D();
    position = new PVector(x, y);
  }

  void run(ArrayList<Boid> boids) {
    flock(boids);
    update();
    render();
  }

  void applyForce(PVector force) {
    // We could add mass here if we want A = F / M
    acceleration.add(force);
  }

  // We accumulate a new acceleration each time based on three rules
  void flock(ArrayList<Boid> boids) {
    PVector wan = wander();          // wander
    PVector sep = separate(boids);   // Separation
    PVector ali = align(boids);      // Alignment
    PVector coh = cohesion(boids);   // Cohesion
    PVector bor = borders();         // borders
    // Arbitrarily weight these forces
    wan.mult(WAN_FACTOR);
    sep.mult(SEP_FACTOR);
    ali.mult(ALI_FACTOR);
    coh.mult(COH_FACTOR);
    bor.mult(BOR_FACTOR);
    // Add the force vectors to acceleration
    applyForce(wan);
    applyForce(sep);
    applyForce(ali);
    applyForce(coh);
    applyForce(bor);
  }

  // Method to update position
  void update() {
    // Update velocity
    velocity.add(acceleration);
    // Limit speed
    velocity.limit(MAX_SPEED);
    position.add(velocity); //<>//
    // Reset accelertion to 0 each cycle
    acceleration.mult(0);
  }

  // A method that calculates and applies a steering force towards a target
  // STEER = DESIRED MINUS VELOCITY
  PVector seek(PVector target) {
    PVector desired = PVector.sub(target, position);  // A vector pointing from the position to the target
    // Scale to maximum speed
    desired.normalize();
    desired.mult(MAX_SPEED);

    // Above two lines of code below could be condensed with new PVector setMag() method
    // Not using this method until Processing.js catches up
    // desired.setMag(MAX_SPEED);

    // Steering = Desired minus Velocity
    PVector steer = PVector.sub(desired, velocity);
    steer.limit(MAX_FORCE);  // Limit to maximum steering force
    return steer;
  }

  void render() {
    // Draw a triangle rotated in the direction of velocity
    float theta = velocity.heading() + radians(90);
    
    fill(200, 100);
    stroke(BOID_COLOUR);
    fill(BOID_COLOUR);
    pushMatrix();
    translate(position.x, position.y);
    rotate(theta);
    beginShape(TRIANGLES);
    vertex(0, -BOID_SIZE*2);
    vertex(-BOID_SIZE, BOID_SIZE*2);
    vertex(BOID_SIZE, BOID_SIZE*2);
    endShape();
    popMatrix();
  }

  PVector wander() {
    PVector wander = PVector.random2D();
    wander.normalize();
    return wander;
  }

  // Wraparound
  PVector borders() {
    float centre_x = width/2;
    float centre_y = height/2;
    PVector centre = new PVector(centre_x, centre_y);
    PVector from_centre = PVector.sub(position, centre);
    if (from_centre.mag() == 0) {
      return from_centre; // return 0 force if at centre
    }
    float bound_radius = height/2;
    PVector nearest_bound_from_centre = from_centre.copy();
    nearest_bound_from_centre.setMag(bound_radius);
    if (from_centre.mag() > nearest_bound_from_centre.mag()) {
      return PVector.mult(nearest_bound_from_centre, -1);
    }
    PVector bound_diff = PVector.sub(from_centre, nearest_bound_from_centre);
    float bound_dist = Float.max(from_centre.dist(nearest_bound_from_centre), EPSILON);
    bound_diff.normalize();
    bound_diff.div(bound_dist);
    return bound_diff;
  }

  // Separation
  // Method checks for nearby boids and steers away
  PVector separate (ArrayList<Boid> boids) {
    float desiredseparation = 25.0f;
    PVector steer = new PVector(0, 0);
    int count = 0;
    // For every boid in the system, check if it's too close
    for (Boid other : boids) {
      float d = PVector.dist(position, other.position);
      // If the distance is greater than 0 and less than an arbitrary amount (0 when you are yourself)
      if ((d > 0) && (d < desiredseparation)) {
        // Calculate vector pointing away from neighbor
        PVector diff = PVector.sub(position, other.position);
        diff.normalize();
        diff.div(d);        // Weight by distance
        steer.add(diff);
        count++;            // Keep track of how many
      }
    }
    // Average -- divide by how many
    if (count > 0) {
      steer.div((float)count);
    }

    // As long as the vector is greater than 0
    if (steer.mag() > 0) {
      // First two lines of code below could be condensed with new PVector setMag() method
      // Not using this method until Processing.js catches up
      // steer.setMag(MAX_SPEED);

      // Implement Reynolds: Steering = Desired - Velocity
      steer.normalize();
      steer.mult(MAX_SPEED);
      steer.sub(velocity);
      steer.limit(MAX_FORCE);
    }
    return steer;
  }

  // Alignment
  // For every nearby boid in the system, calculate the average velocity
  PVector align (ArrayList<Boid> boids) {
    float neighbordist = 50;
    PVector sum = new PVector(0, 0);
    int count = 0;
    for (Boid other : boids) {
      float d = PVector.dist(position, other.position);
      if ((d > 0) && (d < neighbordist)) {
        sum.add(other.velocity);
        count++;
      }
    }
    if (count > 0) {
      sum.div((float)count);
      // First two lines of code below could be condensed with new PVector setMag() method
      // Not using this method until Processing.js catches up
      // sum.setMag(MAX_SPEED);

      // Implement Reynolds: Steering = Desired - Velocity
      sum.normalize();
      sum.mult(MAX_SPEED);
      PVector steer = PVector.sub(sum, velocity);
      steer.limit(MAX_FORCE);
      return steer;
    } 
    else {
      return new PVector(0, 0);
    }
  }

  // Cohesion
  // For the average position (i.e. center) of all nearby boids, calculate steering vector towards that position
  PVector cohesion (ArrayList<Boid> boids) {
    float neighbordist = 50;
    PVector sum = new PVector(0, 0);   // Start with empty vector to accumulate all positions
    int count = 0;
    for (Boid other : boids) {
      float d = PVector.dist(position, other.position);
      if ((d > 0) && (d < neighbordist)) {
        sum.add(other.position); // Add position
        count++;
      }
    }
    if (count > 0) {
      sum.div(count);
      return seek(sum);  // Steer towards the position
    } 
    else {
      return new PVector(0, 0);
    }
  }
}
