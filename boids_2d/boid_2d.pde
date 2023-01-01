int BOID_COLOUR = 255;
float BOID_SIZE = 2.0;

public class Boid
{
  public PVector position;
  public float theta;

  public Boid(float x, float y, float theta)
  {
    this.position = new PVector(x, y);
    this.theta = theta;
  }

  void Render() {
    // Draw a triangle rotated in the direction of velocity
    fill(200, 100);
    stroke(BOID_COLOUR);
    fill(BOID_COLOUR);
    pushMatrix();
    translate(position.x, position.y);
    rotate(radians(180)-theta);
    beginShape(TRIANGLES);
    vertex(0, -BOID_SIZE*2);
    vertex(-BOID_SIZE, BOID_SIZE*2);
    vertex(BOID_SIZE, BOID_SIZE*2);
    endShape();
    popMatrix();
  }
}
