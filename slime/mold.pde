abstract class f_xj
{
  abstract double func(PVector pos);
}


class f34 extends f_xj //Rastrigins function 2.5        f(x)=0  @x=(0,0,...)     -5.12<x[i]<5.12
{
  public double func(PVector pos) 
  {
    double ff=0;
    ff += pos.x*pos.x-10*Math.cos(2.0*Math.PI*pos.x);
    ff += pos.y*pos.y-10*Math.cos(2.0*Math.PI*pos.y);
    int n = 2; // num dimensions
    return ff+10*n;
  }
}


public class Mold
{
  Agent[] Agents;
  PVector[] weight;
  double[] AllFitness;
  double[] SmellOrder;
  int[] SmellIndex;
  PVector bestPositions;
  double worstFitness;
  double bestFitness;
  double Destination_fitness;
  double focus;
  double a;
  double z;
  double p;
  double S;
  double eps;
  f_xj ff;
  int NumAgents;

	public Mold(int iNumAgents)
  {
  	ff= new f34();
  	NumAgents = iNumAgents;
    weight=new PVector[NumAgents];
    Agents=new Agent[NumAgents];
    AllFitness=new double[NumAgents];
    SmellOrder=new double[NumAgents];
    SmellIndex=new int[NumAgents];
    bestPositions=new PVector(0, 0);
  	z=0.03;
    focus = 0;
    eps=0.000000000000000000001;
    Destination_fitness=1E+200;
    
    init();
  }

  void init()
  {
	  for(int i=0;i<NumAgents;i++)
	  {
      Agents[i] = new Agent();
	 	  AllFitness[i]=ff.func(Agents[i].Position);
	  }     
	}

	double[][] sort_and_index(double[] A)
	{
		ArrayList<Double> B=new ArrayList<Double>(); 	
		for(int i=0;i<A.length;i++)
		{
      B.add(A[i]);
    }	
		ArrayList<Double> nstore=new ArrayList<Double>(B);
		Collections.sort(B);
		double[] ret=new double[B.size()];
		Iterator<Double> iterator=B.iterator();
		int ii=0;
		while(iterator.hasNext())
		{
      ret[ii]=iterator.next().doubleValue();
      ii++;
    }
		int[] indexes=new int[B.size()];
		for(int n=0;n<B.size();n++)
		{
      // TODO check this makes sense???
      indexes[n]=nstore.indexOf(B.get(n));
    }
		double[][] outt=new double[2][B.size()];
		for(int i=0;i<B.size();i++)
		{
      outt[0][i]=ret[i];
      outt[1][i]=indexes[i];
    }
		return outt;
	}
    
	
  PVector unifrnd(double a1, double a2)
  { 
    return new PVector(random(a1, a2), random(a1, a2));
	}

	double atanh(double x)
	{
    return 0.5*Math.log( (x + 1.0) / (x - 1.0));
  }
	
	void run()
	{
		for(int i=0;i<NumAgents;i++)
		{
      Agents[i].Render();
	 	  AllFitness[i]=ff.func(Agents[i].Position);
	  }

	  double[][] Smell=sort_and_index(AllFitness);
	   		
	  for(int i=0;i<NumAgents;i++)
	  {
		  SmellOrder[i]=Smell[0][i];
      SmellIndex[i]=(int)Smell[1][i];
		} 
	  worstFitness=SmellOrder[NumAgents-1];
	  bestFitness=SmellOrder[0];
	  S=bestFitness-worstFitness+eps;			
			
	  for(int i=0;i<NumAgents;i++)
	  {
      // TODO check if should be in order of fitness rank rather than arbitrary order
			if(i<(NumAgents/2))
			{
				weight[SmellIndex[i]].x=1.0+Math.random()*Math.log10(((bestFitness-SmellOrder[i])/S)+1.0);
        weight[SmellIndex[i]].y=1.0+Math.random()*Math.log10(((bestFitness-SmellOrder[i])/S)+1.0);
		  }
			else
			{
				weight[SmellIndex[i]].x=1.0-Math.random()*Math.log10(((bestFitness-SmellOrder[i])/S)+1.0);
        weight[SmellIndex[i]].y=1.0-Math.random()*Math.log10(((bestFitness-SmellOrder[i])/S)+1.0);
		  }  
		}		
			
		if(bestFitness < Destination_fitness)
		{
      bestPositions.x=Agents[SmellIndex[0]].Position.x;
      bestPositions.y=Agents[SmellIndex[0]].Position.y;
      
			Destination_fitness=bestFitness;
		}
			
		a=atanh(1.0-focus+eps);
			
		for(int i=0;i<NumAgents;i++)
		{
		  if(Math.random()<z)
			{
        Agents[i].RandomizePosition();
			}	
				
			p=Math.tanh(Math.abs(AllFitness[i]-Destination_fitness));
			PVector vb=unifrnd(-a,a);
			PVector vc=1.0-focus;
			
      Agents[i].ReweightPosition(NumAgents, p, vb, vc, Agents, weight, bestPositions);
      Agents[i].EnsureInBounds();
		}
  }
}
