
PVector getOrthoVecA(PVector vec) {
  PVector vecA = new PVector(0., 0.);
  if (vec.x != 0) {
    vecA.y = vec.x;
    vecA.x = -1 * vec.y;
  } else if (vec.y != 0) {
    vecA.z = vec.y;
    vecA.y = -1 * vec.z;
  } else {
    vecA.x = vec.z;
    vecA.z = -1 * vec.x;
  }
  vecA.normalize();
  return vecA;
}

PVector getOrthoVecB(PVector vec) {
  PVector vecB = new PVector(0., 0.);
  if (vec.x != 0) {
    vecB.z = vec.x;
    vecB.x = -1 * vec.z;
  } else if (vec.y != 0) {
    vecB.x = vec.y;
    vecB.y = -1 * vec.x;
  } else {
    vecB.y = vec.z;
    vecB.z = -1 * vec.y;
  }
  vecB.normalize();
  return vecB;
}


void setup() {
  size(1200, 600);
}

void draw() {
  background(0);
  
  PVector pos = new PVector(400, 300, 0);
  PVector vel = new PVector(40, 0, 0);
  float sensorDist = 80;
  float sensorOffset = 60;
  float sensorOffDist = sensorOffset / tan(asin(sensorOffset / sensorDist));
  
  PVector velNorm = vel.copy();
  velNorm.normalize();
  PVector vecAhead = PVector.mult(velNorm, sensorDist);
  
  PVector vecX = getOrthoVecA(vel);
  PVector vecY = getOrthoVecB(vel);

  PVector vecA = PVector.add(PVector.mult(velNorm, sensorOffDist), PVector.mult(vecX, sensorOffset));
  PVector vecB = PVector.add(PVector.mult(velNorm, sensorOffDist), PVector.mult(vecX, -1 * sensorOffset));
  PVector vecC = PVector.add(PVector.mult(velNorm, sensorOffDist), PVector.mult(vecY, sensorOffset));
  PVector vecD = PVector.add(PVector.mult(velNorm, sensorOffDist), PVector.mult(vecY, -1 * sensorOffset));
  
  PVector senseAhead = PVector.add(pos, vecAhead);
  PVector senseA = PVector.add(pos, vecA);
  PVector senseB = PVector.add(pos, vecB);
  PVector senseC = PVector.add(pos, vecC);
  PVector senseD = PVector.add(pos, vecD);
  
  stroke(255);
  circle(pos.x, pos.y, 10);
  line(pos.x, pos.y, pos.x + vel.x, pos.y + vel.y);
  
  circle(senseAhead.x, senseAhead.y, 5);
  circle(senseA.x + senseA.z * 0.1, senseA.y, 5);
  circle(senseB.x + senseB.z * 0.1, senseB.y, 5);
  circle(senseC.x + senseC.z * 0.1, senseC.y, 5);
  circle(senseD.x + senseD.z * 0.1, senseD.y, 5);
  
  noFill();
  circle(pos.x, pos.y, sensorDist * 2);
}
