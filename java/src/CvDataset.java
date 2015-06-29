import java.util.ArrayList;

import weka.core.Instances;


public class CvDataset {
	private Instances train;
	private ArrayList<Instances> test;
	
	public CvDataset()
	{
		test = new ArrayList<Instances>();
	}
	
	
	public int getNumTestSets()
	{
		return test.size();
	}
	
	public Instances getTrainingset()
	{
		return train;
	}
	
	public Instances getTestset(int index)
	{
		if(index > getNumTestSets() || index < 0)
		{
			throw new IndexOutOfBoundsException("index " + index + " not in range of [0-" + (getNumTestSets()-1) + "]");
		}
		return test.get(index);
	}
	
	public void setTrainingSet(Instances training)
	{
		train = training;
	}
	
	public void addTestset(Instances testset)
	{
		test.add(testset);
	}
	
	public void resetTestsets()
	{
		test.clear();
	}
	
	public void setTestsets(ArrayList<Instances> testsets)
	{
		test = testsets;
	}
}
