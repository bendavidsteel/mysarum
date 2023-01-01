
public class Neighbour
{
  public PVector position;
  public float dist;
  public float theta;
  public float neighbourTheta;
  public int idx;

  public Neighbour(PVector position, float dist, float theta, float neighbourTheta, int i)
  {
    this.position = position;
    this.dist = dist;
    this.theta = theta;
    this.neighbourTheta = neighbourTheta;
    this.idx = i;
  }
}

public class Flock
{
  int amount;
  float speed;
  Boid[] boids;
  float turnSpeed;
  float detectDist;
  int numTrack;

  public Flock(int amount)
  {
    this.amount = amount;
    speed = 2.5;
    boids = new Boid[amount];
    turnSpeed = 1 / 32;
    detectDist = 100;
    numTrack = 3;
    Init();
  }
      
  void Init() {
    for (int i = 0; i < amount; i++)
    {
      float x = random(0, width);
      float y = random(0, height);
      float theta = random(0, (float) Math.PI * 2);
      boids[i] = new Boid(x, y, theta);
    }
  }

  void Run() {
    for (var i = 0; i < this.amount; i++) {
      Boid boid = boids[i];

      float velX = (float) (this.speed * Math.sin(boid.theta));
      float velY = (float) (this.speed * Math.cos(boid.theta));
      //float normBackX = boid.position.x - ((velX / this.speed) * 5);
      //float normBackY = boid.position.y - ((velY / this.speed) * 5);

      boid.theta = normTheta(random((float) (boid.theta - (Math.PI * turnSpeed)), (float) (boid.theta + (Math.PI * turnSpeed))));
      
      ArrayList<Neighbour> neighbourFlakes = rankNeighbours(boid, boids, detectDist);
      // only checking closest
      for (int m = 0; m < Math.min(numTrack, neighbourFlakes.size()); m++) {
        Neighbour neighbourFlake = neighbourFlakes.get(m);
        float avoidTheta = getAvoidTheta(boid, neighbourFlake);
        boid.theta += avoidTheta / 1.2;

        // turn in same direction as neighbours
        float flakeThetaDiff = getThetaDiff(boid.theta, neighbourFlake.neighbourTheta);
        boid.theta += flakeThetaDiff / 5;
      }

      // turn in direction of centroid
      float centroidX = 0;
      float centroidY = 0;
      for (int m = 0; m < neighbourFlakes.size(); m++) {
        centroidX += neighbourFlakes.get(m).position.x / neighbourFlakes.size();
        centroidY += neighbourFlakes.get(m).position.y / neighbourFlakes.size();
      }
      float centroidTheta = thetaBetweenPoints(centroidX - boid.position.x, centroidY - boid.position.y);
      float centroidThetaDiff = getThetaDiff(boid.theta, centroidTheta);
      boid.theta += centroidThetaDiff / 30;

      boid.theta = normTheta(boid.theta);
      velX = (float) (this.speed * Math.sin(boid.theta));
      velY = (float) (this.speed * Math.cos(boid.theta));
      // invert y because origin is at top left
      boid.position.y += velY;
      boid.position.x += velX;
      checkReset(boid);
       //<>//
      boid.Render();
    }
  }

  void checkReset(Boid boid) {
    if (boid.position.x > width) {
        boid.position.x = 0;
    }
    else if (boid.position.x < 0) {
        boid.position.x = width;
    }
    if (boid.position.y > height) {
        boid.position.y = 0;
    }
    else if (boid.position.y < 0) {
        boid.position.y = height;
    }
  }

  float normTheta(float theta) {
    while (theta < 0) {
        theta += 2 * Math.PI;
    }
    while (theta >= (2 * Math.PI)) {
        theta -= 2 * Math.PI;
    }
    return theta;
  }

  float thetaBetweenPoints(float x, float y) {
    float theta = (float) (-Math.atan2(y, x) + (Math.PI / 2));
    float normedTheta = normTheta(theta);
    return normedTheta;
  }

  float objTheta(Boid objA, Boid objB) {
    return thetaBetweenPoints(objB.position.x - objA.position.x, objB.position.y - objA.position.y);
  }

  float objDist(Boid objA, Boid objB) {
    return (float) Math.sqrt(Math.pow(objB.position.x - objA.position.x, 2) + Math.pow(objB.position.y - objA.position.y, 2));
  }

  ArrayList<Neighbour> rankNeighbours(Boid boid, Boid[] boids, float detectDist) {
    ArrayList<Neighbour> neighbours = new ArrayList<Neighbour>();
    for (int i = 0; i < boids.length; i++) {
      Boid boidNeighbour = boids[i];
      float dist = objDist(boid, boidNeighbour);
      if (dist < detectDist) {
        float theta = objTheta(boid, boidNeighbour);
        neighbours.add(new Neighbour(boidNeighbour.position, dist, theta, boidNeighbour.theta, i));
      }
    }
    neighbours.sort((a, b) -> Float.compare(a.dist, b.dist));
    return neighbours;
  }

  float getThetaDiff(float thetaA, float thetaB) {
    if (thetaB > thetaA) {
        if (thetaB < thetaA + Math.PI) {
            return thetaB - thetaA;
        }
        else if (thetaB > thetaA + Math.PI) {
            return (float) -(thetaA + ((2 * Math.PI) - thetaB));
        }
        else {
            return (float) Math.PI;
        }
    }
    else if (thetaB < thetaA) {
        if (thetaB > thetaA - Math.PI) {
            return thetaB - thetaA;
        }
        else if (thetaB < thetaA - Math.PI) {
            return (float) (thetaB + ((2 * Math.PI) - thetaA));
        }
        else {
            return (float) Math.PI;
        }
    }
    else {
        return 0;
    }
  }

  float getAvoidTheta(Boid boid, Neighbour obj) {
    // if already collided just carry on
    if (obj.dist == 0) {
        return 0;
    }

    float thetaDiff = getThetaDiff(boid.theta, obj.theta);

    // check if in sight
    float absThetaDiff = Math.abs(thetaDiff);
    float avoidDir = 0;
    if (thetaDiff == 0) {
        avoidDir = Math.random() < 0.5 ? 1 : -1;
    }
    else if (absThetaDiff < (Math.PI / 2)) {
        // weight by angle
        // is 1 or 
        avoidDir = (-1 * thetaDiff) / absThetaDiff;
    }
    return (float) (avoidDir * (Math.PI / Math.min(obj.dist, 8)) * Math.cos(thetaDiff / 2));
  }
}
