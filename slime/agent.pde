
int BOID_COLOUR = 255;
float BOID_SIZE = 2.0;

public class Agent
{
  PVector Position;
  
  public Agent()
  {
    RandomizePosition();
  }
  
  void RandomizePosition()
  {
    Position = new PVector(random(0, width), random(0, height));
  }
  
  void EnsureInBounds()
  {
    if ((Position.x < 0) || (Position.x > width) || (Position.y < 0) || (Position.y > height))
    {
      RandomizePosition();
    }
  }
  
  int RandInt(int NumAgents)
  {
    return (int)(Math.floor(Math.random()*(double)NumAgents));
  }
  
  void ReweightPosition(int NumAgents, double p, PVector vb, PVector vc, Agent Agents[], PVector weight[], PVector bestPositions)
  {
    r=Math.random(); 
    A=RandInt(NumAgents); 
    B=RandInt(NumAgents);  
    if(r<p)
    {
      Position.x=bestPositions.x+vb.x*(weight[i].x*Agents[A].Position.x-Agents[B].Position.x);
      Position.y=bestPositions.y+vb.y*(weight[i].y*Agents[A].Position.y-Agents[B].Position.y);  
    }
    else
    {
      Position.x=vc.x*Position.x;
      Position.y=vc.y*Position.y;
    }
  }
  
  void Render()
  {
    fill(200, 100);
    stroke(BOID_COLOUR);
    fill(BOID_COLOUR);
    pushMatrix();
    translate(Position.x, Position.y);
    beginShape(TRIANGLES);
    vertex(0, -BOID_SIZE);
    vertex(-BOID_SIZE, BOID_SIZE);
    vertex(BOID_SIZE, BOID_SIZE);
    endShape();
    popMatrix();
  }
}
