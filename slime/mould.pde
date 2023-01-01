

public class Mould
{
  float MoveSpeed;
  float TurnSpeed;

  float SensorAngleDegrees;
  float SensorOffsetDst;
  int SensorSize;
  float[] Colour;
  
  public Mould(float moveSpeed, float turnSpeed, float sensorAngleDegrees, float sensorOffsetDst, int sensorSize, float[] colour)
  {
    this.MoveSpeed = moveSpeed;
    this.TurnSpeed = turnSpeed;
    this.SensorAngleDegrees = sensorAngleDegrees;
    this.SensorOffsetDst = sensorOffsetDst;
    this.SensorSize = sensorSize;
    this.Colour = colour;
  }
};
