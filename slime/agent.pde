
int BOID_COLOUR = 255;
float BOID_SIZE = 2.0;

enum AgentSpawn {
  RANDOM,
  CENTRE,
}


public class Agent
{
  PVector Position;
  float Angle;
  float[] SpeciesMask;
  int SpeciesIndex;
  
  public Agent(int species, int numSpecies, int mapHeight, int mapWidth, AgentSpawn spawnStrategy)
  {
    Spawn(mapHeight, mapWidth, spawnStrategy);
    SpeciesIndex = species;
    SpeciesMask = new float[numSpecies];
    for (int i = 0; i < numSpecies; i++)
    {
      if (species == i)
      {
        SpeciesMask[i] = 1.0;
      }
      else
      {
        SpeciesMask[i] = 0.0;
      }
    }
  }
  
  void Spawn(int mapHeight, int mapWidth, AgentSpawn spawnStrategy)
  {
    if (spawnStrategy == AgentSpawn.RANDOM)
    {
      Position = new PVector(random(0, mapWidth), random(0, mapHeight));
    }
    else if (spawnStrategy == AgentSpawn.CENTRE)
    {
      Position = new PVector(mapWidth / 2, mapHeight / 2);
    }
    Angle = random(0.0, 2 * (float) Math.PI);
  }
  
  void EnsureRebound(int mapHeight, int mapWidth)
  {
    PVector velocity = GetVelocity();
    PVector normal = new PVector();
    boolean rebound = false;
    if (Position.x < 0)
    {
      normal = new PVector(1, 0);
      rebound = true;
    }
    if (Position.x >= mapWidth)
    {
      normal = new PVector(-1, 0);
      rebound = true;
    }
    if (Position.y < 0)
    {
      normal = new PVector(0, 1);
      rebound = true;
    }
    if (Position.y >= mapHeight)
    {
      normal = new PVector(0, -1);
      rebound = true;
    }
    
    if (rebound)
    {
      PVector newVelocity = PVector.sub(velocity, PVector.mult(normal, 2 * PVector.dot(velocity, normal)));
      float newAngle = atan2(newVelocity.y, newVelocity.x);
      Rebound(mapHeight, mapWidth, newAngle);
    }
  }
  
  void Rebound(int mapHeight, int mapWidth, float randomAngle)
  {
    Position.x = min(mapWidth-1, max(0, Position.x));
    Position.y = min(mapHeight-1, max(0, Position.y));
    Angle = randomAngle;
  }
  
  PVector GetVelocity()
  {
    return new PVector(cos(Angle), sin(Angle));
  }
  
  void UpdatePosition(float deltaTime, float moveSpeed)
  {
    PVector direction = GetVelocity();
    Position = PVector.add(Position, PVector.mult(direction, deltaTime * moveSpeed));
  }
}
