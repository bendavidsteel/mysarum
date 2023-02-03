
public class Moulds
{
  Mould[] SpeciesSettings;
  int NumSpecies;
  
  ArrayList<Agent> Agents;
  int NumAgents;
  
  int PixelScale;
  int MapWidth;
  int MapHeight;
  
  float[][][] TrailMap;
  
  float DeltaTime;
  
  float DecayRate;
  float DiffuseRate;
  float DeathRate;
  
  public Moulds(int NumAgents, int PixelScale)
  {
    this.NumAgents = NumAgents;
    int trailWeight = 5;
    DecayRate = 0.999;
    DeathRate = 1.0;
    DiffuseRate = 0.01;
    DeltaTime = 1.0;
    
    this.PixelScale = PixelScale;
    MapWidth = width / PixelScale;
    MapHeight = height / PixelScale;
    
    NumSpecies = 3;
    SpeciesSettings = new Mould[NumSpecies];
    // species 1
    float[] speciesColour1 = new float[] {57, 255, 20, 0};
    float[] speciesColour2 = new float[] {0, 192, 249, 0};
    float[] speciesColour3 = new float[] {188, 0, 254, 0};
    float moveSpeed = 0.8;
    float turnSpeed = 0.2;
    float sensorAngleDegrees = 60;
    float sensorOffsetDist = 30;
    int sensorSize = 1;
    SpeciesSettings[0] = new Mould(moveSpeed, turnSpeed, sensorAngleDegrees, sensorOffsetDist, sensorSize, speciesColour1);
    SpeciesSettings[1] = new Mould(moveSpeed, turnSpeed*2, sensorAngleDegrees/2, sensorOffsetDist/2, sensorSize, speciesColour2);
    SpeciesSettings[2] = new Mould(moveSpeed, turnSpeed*1.5, sensorAngleDegrees/1.5, sensorOffsetDist/1.5, sensorSize, speciesColour3);
    
    int species1 = 0;
    AgentSpawn spawnStrategy1 = AgentSpawn.CENTRE;
    Agents = new ArrayList<Agent>();
    for (int i = 0; i < NumAgents/3; i++)
    {
      Agents.add(new Agent(species1, NumSpecies, MapHeight, MapWidth, spawnStrategy1, trailWeight));
    }
    
    int species2 = 1;
    AgentSpawn spawnStrategy2 = AgentSpawn.RANDOM;
    for (int i = NumAgents/3; i < 2 * NumAgents/3; i++)
    {
      Agents.add(new Agent(species2, NumSpecies, MapHeight, MapWidth, spawnStrategy2, trailWeight));
    }
    
    int species3 = 2;
    for (int i = 2*NumAgents/3; i < NumAgents; i++)
    {
      Agents.add(new Agent(species3, NumSpecies, MapHeight, MapWidth, spawnStrategy2, trailWeight));
    }
    
    // initialize trail maps
    TrailMap = new float[MapHeight][MapWidth][NumSpecies];
  }
  
  float[] Mult(float[] vec, float sca)
  {
    float[] mult = new float[vec.length];
    for (int i = 0; i < vec.length; i++)
    {
      mult[i] = vec[i] * sca;
    }
    return mult;
  }
  
  float[] Div(float[] vec, float sca)
  {
    float[] res = new float[vec.length];
    for (int i = 0; i < vec.length; i++)
    {
      res[i] = vec[i] / sca;
    }
    return res;
  }
  
  float[] Sub(float[] vec, float sca)
  {
    float[] mult = new float[vec.length];
    for (int i = 0; i < vec.length; i++)
    {
      mult[i] = vec[i] - sca;
    }
    return mult;
  }
  
  float[] Add(float[] a, float[] b)
  {
    float[] res = new float[a.length];
    for (int i = 0; i < a.length; i++)
    {
      res[i] = a[i] + b[i];
    }
    return res;
  }
  
  float[] Min(float[] vec, float sca)
  {
    float[] res = new float[vec.length];
    for (int i = 0; i < vec.length; i++)
    {
      res[i] = min(vec[i], sca);
    }
    return res;
  }
  
  float[] Max(float[] vec, float sca)
  {
    float[] res = new float[vec.length];
    for (int i = 0; i < vec.length; i++)
    {
      res[i] = max(vec[i], sca);
    }
    return res;
  }
  
  float Dot(float[] a, float[] b)
  {
    float res = 0;
    for (int i = 0; i < a.length; i++)
    {
      res += a[i] * b[i];
    }
    return res;
  }
  
  float[] IntArrayToFloatArray(int[] a)
  {
    float[] res = new float[a.length];
    for (int i = 0; i < a.length; i++)
    {
      res[i] = (float) a[i];
    }
    return res;
  }
  
  float[] InitArray(int len, float init)
  {
    float[] res = new float[len];
    for (int i = 0; i < len; i++)
    {
      res[i] = init;
    }
    return res;
  }
  
  float Clamp(float val, float a, float b)
  {
    return max(min(val, b), a);
  }
  
  float[] ToOneHot(int idx, int num)
  {
    float[] res = new float[num];
    for (int i = 0; i < num; i++)
    {
      if (idx == i)
      {
        res[i] = 1.0;
      }
      else
      {
        res[i] = 0.0;
      }
    }
    return res;
  }
  
  float Sense(Agent agent, Mould settings, float sensorAngleOffset) {
    float sensorAngle = agent.Angle + sensorAngleOffset;
    PVector sensorDir = new PVector(cos(sensorAngle), sin(sensorAngle));
  
    PVector sensorPos = PVector.add(agent.Position, PVector.mult(sensorDir, settings.SensorOffsetDst));
    int sensorCentreX = (int) sensorPos.x;
    int sensorCentreY = (int) sensorPos.y;
  
    float sum = 0;
  
    float[] senseWeight = Sub(Mult(agent.SpeciesMask, 2.0), 1.0);
  
    for (int offsetX = -settings.SensorSize; offsetX <= settings.SensorSize; offsetX ++) {
      for (int offsetY = -settings.SensorSize; offsetY <= settings.SensorSize; offsetY ++) {
        int sampleX = min(MapWidth - 1, max(0, sensorCentreX + offsetX));
        int sampleY = min(MapHeight - 1, max(0, sensorCentreY + offsetY));
        sum += Dot(senseWeight, TrailMap[sampleY][sampleX]);
      }
    }
  
    return sum;
  }
  
  void Update ()
  {
    for (Agent agent : Agents)
    { 
      // reduce trail weight
      agent.TrailWeight *= DeathRate;
      
      Mould species = SpeciesSettings[agent.SpeciesIndex];
    
      // Steer based on sensory data
      float sensorAngleRad = radians(species.SensorAngleDegrees); //<>//
      float weightForward = Sense(agent, species, 0);
      float weightLeft = Sense(agent, species, sensorAngleRad);
      float weightRight = Sense(agent, species, -sensorAngleRad);
    
      
      float randomSteerStrength = random(0, 1);
      float turnSpeed = species.TurnSpeed * 2 * (float) Math.PI;
    
      // Continue in same direction
      if ((weightForward > weightLeft) && (weightForward > weightRight)) {
        agent.Angle += 0;
      }
      else if (weightForward < weightLeft && weightForward < weightRight) { // TODO does this make sense?
        agent.Angle += (randomSteerStrength - 0.5) * 2 * turnSpeed * DeltaTime;
      }
      // Turn right
      else if (weightRight > weightLeft) {
        agent.Angle -= randomSteerStrength * turnSpeed * DeltaTime;
      }
      // Turn left
      else if (weightLeft > weightRight) {
        agent.Angle += randomSteerStrength * turnSpeed * DeltaTime;
      }
    
      // Update position
      agent.UpdatePosition(DeltaTime, species.MoveSpeed);
      
      // Clamp position to map boundaries, and pick new random move dir if hit boundary
      agent.EnsureRebound(MapHeight, MapWidth);
    
      int x = int(agent.Position.x);
      int y = int(agent.Position.y);
      float[] oldTrail = TrailMap[y][x];
      TrailMap[y][x] = Min(Add(oldTrail, Mult(agent.SpeciesMask, agent.TrailWeight * DeltaTime)), 1.0);
    } //<>//

    Agents.removeIf(a -> (a.TrailWeight < 0.01));
    
    Diffuse();
    UpdateColourMap(); //<>//
  }
  
  
  
  void Diffuse ()
  {
    for (int y = 0; y < MapHeight; y++)
    {
      for (int x = 0; x < MapWidth; x++)
      {
        float[] sum = InitArray(NumSpecies, 0);
        float[] originalCol = TrailMap[y][x];
        // 3x3 blur
        for (int offsetX = -1; offsetX <= 1; offsetX ++) {
          for (int offsetY = -1; offsetY <= 1; offsetY ++) {
            int sampleX = min(MapWidth-1, max(0, x + offsetX));
            int sampleY = min(MapHeight-1, max(0, y + offsetY));
            sum = Add(sum, TrailMap[sampleY][sampleX]);
          }
        }
      
        float[] blurredCol = Div(sum, 9);
        float diffuseWeight = Clamp(DiffuseRate * DeltaTime, 0, 1);
        blurredCol = Add(Mult(originalCol,  (1 - diffuseWeight)), Mult(blurredCol, diffuseWeight));
      
        //DiffusedTrailMap[id.xy] = blurredCol * saturate(1 - decayRate * deltaTime);
        TrailMap[y][x] = Max(Mult(blurredCol, DecayRate * DeltaTime), 0.0);
      }
    }
  }
  
  void UpdateColourMap ()
  {
    loadPixels();
    for (int y = 0; y < MapHeight; y++)
    {
      for (int x = 0; x < MapWidth; x++)
      {
        float[] map = TrailMap[y][x];
      
        float[] colour = InitArray(3, 0);
        for (int i = 0; i < NumSpecies; i ++) {
          float[] mask = ToOneHot(i, NumSpecies);
          colour = Add(colour, Mult(SpeciesSettings[i].Colour, Dot(map, mask))); 
        }
        
        for (int i = 0; i < PixelScale; i++)
        {
          for (int j = 0; j < PixelScale; j++)
          {
            pixels[((y * PixelScale) + i) * width + (x * PixelScale) + j] = color(colour[0], colour[1], colour[2]);
          }
        }
      }
    }
    updatePixels();
  }
}
