import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import weka.core.Instances;
import weka.core.SelectedTag;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class ClassificationEvaluator {

	public static int calcPercentage(int sampleSize, Instances dataset)
	{
		int numClasses = dataset.numClasses();
		int numInstances = dataset.numInstances();
		float percentage = 100 * sampleSize * numClasses / (float) numInstances;
		return (int)Math.ceil(percentage);
	}
	
	/** @brief split dataset into train and test sets using numSamplesPerClass */
	public static CvDataset getCvDataset(Instances dataset, int train_numSamplesPerClass, int test_numSamplesPerClass) throws Exception
	{
		int seed = (int) System.currentTimeMillis();
		return getCvDataset(dataset, train_numSamplesPerClass, test_numSamplesPerClass, seed);
	}
	
	
	/**
	 * @brief split dataset into train and test sets using numSamplesPerClass 
	 * @param dataset
	 * @param train_numSamplesPerClass
	 * @param seed
	 * @return
	 * @throws Exception
	 */
	public static CvDataset getCvDataset(Instances dataset, int train_numSamplesPerClass, int test_numSamplesPerClass, int seed) throws Exception
	{
		CvDataset out = new CvDataset();
		
		// Sampling params
		int percentage = calcPercentage(train_numSamplesPerClass, dataset);
		String filterOptions = "-Z " + percentage + " -no-replacement -B 1";
		
		// create ResampleFilter
		Resample resamp = new Resample();
		resamp.setOptions(weka.core.Utils.splitOptions(filterOptions));
		resamp.setRandomSeed(seed);
		resamp.setInputFormat(dataset);
		
		// use the first resample as training set
		Instances trainingset = Resample.useFilter(dataset, resamp);
		out.setTrainingSet(trainingset);
		
		System.out.println(trainingset.size());
		System.out.println("ohne zuruecklegen: " + resamp.getNoReplacement());
		System.out.println("prozent: " + resamp.getSampleSizePercent());
		System.out.println(trainingset.attributeStats(trainingset.numAttributes() -1));
		
		int numTestSets = 1;
		
		Instances remainingDataset = new Instances(dataset);
		for(int i = 0; i < numTestSets; i++)
		{
//			System.out.println("------------\t testset nr: " + (i+1) + "\t-------------");
			
			// get the remaining dataset
			resamp.setInputFormat(dataset);
			resamp.setInvertSelection(true);
			remainingDataset = Filter.useFilter(remainingDataset, resamp);
			
//			System.out.println("remaining Set Size = " + remainingDataset.size());
			
			// resample the remaining dataset for a testset
			resamp.setInvertSelection(false);
			percentage = calcPercentage(test_numSamplesPerClass, remainingDataset);
			resamp.setSampleSizePercent(percentage);
			resamp.setInputFormat(remainingDataset);
			Instances testset = Resample.useFilter(remainingDataset, resamp);
			out.addTestset(testset);
			
//			System.out.println("testset Size: " + testset.numInstances());
//			System.out.println("percent of remaining set: " + percentage);	
//			System.out.println(testset.attributeStats(CLASS_IDX));
		}	
		return out;
	}
	
	/**
	 * @param cfier Classifier to evaluate
	 * @param dataset Dataset to evaluate on
	 * @param num_iterations Number random subsamples of dataset
	 * @return the average percentage of correctly classified instanced throughout the iterations 
	 * @throws Exception
	 */
	public static double[] evaluateClassifier(Classifier cfier, Instances dataset, int num_iterations) throws Exception
	{
		if(num_iterations <= 0)
		{
			throw new IllegalArgumentException("num_itterations must be greater than zero");
		}
		
		final int TRAIN_SAMPLES_PER_CLASS = 15;
		final int TEST_SAMPLES_PER_CLASS = 15;
		double avg_test_correct = 0;
		double avg_train_correct = 0;
		
		for(int iteration = 0; iteration < num_iterations; iteration++)
		{
			//split dataset randomly in train and testset
			CvDataset mySet = getCvDataset(dataset, TRAIN_SAMPLES_PER_CLASS, TEST_SAMPLES_PER_CLASS, iteration);
			
			//train classifier
			cfier.buildClassifier(mySet.getTrainingset());
			
			// evaluate trained classifier
			Evaluation eval = new Evaluation(mySet.getTrainingset());
			eval.evaluateModel(cfier, mySet.getTestset(0));
			avg_test_correct += eval.correct()/eval.numInstances();
//			System.out.println(eval.toSummaryString());
			
			eval.evaluateModel(cfier, mySet.getTrainingset());
			avg_train_correct += eval.correct()/eval.numInstances();
			
		}
		avg_test_correct /= num_iterations;
		avg_train_correct /= num_iterations;
		double[] metrics = new double[2];
		metrics[0] = avg_test_correct;
		metrics[1] = avg_train_correct;
		
//		System.out.println("Test Average correctly classified: " + avg_test_correct);
//		System.out.println("Train Average correctly classified: " + avg_train_correct);
		
		return metrics;
	}
	
	private static Instances readDataset(String filepath)
			throws FileNotFoundException, IOException {
		BufferedReader breader = null;
		breader = new BufferedReader(new FileReader(filepath));
		
		Instances dataset = new Instances (breader);
		final int CLASS_IDX = dataset.numAttributes() -1;
		dataset.setClassIndex(CLASS_IDX);
		breader.close();
		return dataset;
	}

	private static HashMap<String, Classifier> generateClassifiers() {
		HashMap<String, Classifier> cfiersToBeUsed = new HashMap<String, Classifier>();
		final String PARAMETER_SEPERATOR = " ";
		
		// SVM
		HashMap<Integer, String> kernels =  new HashMap<Integer, String>();
		kernels.put(0, "Linear");
		kernels.put(1, "polynomial");
		kernels.put(2, "rbf");
		kernels.put(3, "sigmoid");
		final double GAMMA = 12;
		final double COEF0 = 300;
		final double COST = 2500;
		int kernelType = 1;
		int degree = 5;
		String cfierKey;
		
		createSVM(cfiersToBeUsed, PARAMETER_SEPERATOR, kernels, GAMMA, COEF0,
				COST, kernelType, degree);
		
//		svmParamGridSearch(cfiersToBeUsed, PARAMETER_SEPERATOR, kernels);
		
		
		
		// Random Forest
		final int NUM_TREES = 800;
		RandomForest randForest =  new RandomForest();
		randForest.setNumTrees(NUM_TREES);
		cfierKey = "RandomForest" + PARAMETER_SEPERATOR + "NumTrees:" + NUM_TREES;
//		cfiersToBeUsed.put(cfierKey, randForest);
		
		// KNN
		int k = 15;
		boolean useCrossValidation = true;
		IBk knn = new IBk();
		knn.setCrossValidate(useCrossValidation);
		knn.setKNN(k);
		cfierKey = "knn" + PARAMETER_SEPERATOR + "k:" + k + PARAMETER_SEPERATOR + "useCrossValidation:" + useCrossValidation;
//		cfiersToBeUsed.put(cfierKey, knn);
		return cfiersToBeUsed;
	}

	private static void createSVM(HashMap<String, Classifier> cfiersToBeUsed,
			final String PARAMETER_SEPERATOR, HashMap<Integer, String> kernels,
			final double GAMMA, final double COEF0, final double COST,
			int kernelType, int degree) {
		String cfierKey;
		LibSVM svm1 = new LibSVM();
		svm1.setKernelType(new SelectedTag(kernelType, LibSVM.TAGS_KERNELTYPE));
		svm1.setGamma(GAMMA);
		svm1.setCoef0(COEF0);
		svm1.setCost(COST);
		svm1.setDegree(degree);
		cfierKey = "SVM" + PARAMETER_SEPERATOR + "gamma:" + GAMMA + PARAMETER_SEPERATOR + "Kerneltype:" +
		kernels.get(svm1.getKernelType().getSelectedTag().getID()) + PARAMETER_SEPERATOR + "cost:" + COST + PARAMETER_SEPERATOR + "coef0:" + COEF0;
		cfiersToBeUsed.put(cfierKey, svm1);
	}

	/**
	 * @param cfiersToBeUsed [out] list of svm classifiers with different parameters
	 * @param PARAMETER_SEPERATOR
	 * @param kernels map containing names for kernels with corresponding keys
	 */
	private static void svmParamGridSearch(
			HashMap<String, Classifier> cfiersToBeUsed,
			final String PARAMETER_SEPERATOR, HashMap<Integer, String> kernels) {
		String cfierKey;
		// exclude sigmoid since tests showed bad results
		for(int kernelType = 0; kernelType < 3; kernelType++)
		{
			for(int cost = 1; cost < 602; cost += 300)
			{
				for(int gamma = 0; gamma < 31; gamma += 10)
				{
					for(int coef = 0; coef < 301; coef += 200)
					{
						for(int degree = 2; degree < 7; degree++)
						{
							LibSVM svm1 = new LibSVM();
							svm1.setKernelType(new SelectedTag(kernelType, LibSVM.TAGS_KERNELTYPE));
							svm1.setGamma(gamma);
							svm1.setCoef0(coef);
							svm1.setCost(cost);
							svm1.setDegree(degree);
							cfierKey = "SVM" + PARAMETER_SEPERATOR + "gamma:" + gamma + PARAMETER_SEPERATOR + "Kerneltype:" +
							kernels.get(svm1.getKernelType().getSelectedTag().getID()) + PARAMETER_SEPERATOR + "cost:" + cost + PARAMETER_SEPERATOR + "coef0:" + coef;
							if(kernelType == 1)
							{
								cfierKey += PARAMETER_SEPERATOR + "degree: " + degree;
							}
							cfiersToBeUsed.put(cfierKey, svm1);
							if(kernelType != 1) // only setup svm once if not using polynomal kernal (1)
							{
								break;
							}
						}
						if(!(kernelType == 1 || kernelType == 3))
						{
							break;
						}
					}
					if(kernelType == 0)
					{
						break;
					}
				}
			}
		}
	}
	
	public static void main(String[] args) throws Exception{
		if(args.length == 0)
		{
			System.err.println("No path to datasets given. exiting.");
			return;
		}
		
		final String CLASSIFICATION_RESULT_PATH = "result.csv";
		final File arffFolder = new File(args[0]);
		
		if(!arffFolder.exists())
		{
			System.err.println("Given path: " + args[0] + "does not exist. Exiting.");
			return;
		}
		
		// Objects to store results
		TreeMap<Double, String> sortedResults = new TreeMap<Double, String>();
		FileWriter fwriter = new FileWriter(CLASSIFICATION_RESULT_PATH);
		BufferedWriter resultCsvWriter = new BufferedWriter(fwriter);
		resultCsvWriter.write("Feature,Classifier,PercentageCorrectTest,PercentageCorrectTrain\n");
		
		// define Classifiers
		HashMap<String, Classifier> cfiersToBeUsed = generateClassifiers();
		
		// run trough datasets
		for(final File fileEntry : arffFolder.listFiles())
		{
			// reading dataset
			System.out.println("reading: " + fileEntry.getName());
			Instances dataset = readDataset(fileEntry.getPath());
			System.out.println("NumInstances = " + dataset.size());
			
			Iterator<Map.Entry<String,Classifier>> iter = cfiersToBeUsed.entrySet().iterator();
			while(iter.hasNext())
			{
				Map.Entry<String,Classifier> entry = iter.next();
				Classifier cfier = entry.getValue();
				int numIterations = 1;
				double[] results = evaluateClassifier(cfier, dataset, numIterations);
				double testCorrect = results[0];
				double trainCorrect = results[1];
				resultCsvWriter.write(fileEntry.getName() + "," + entry.getKey() + "," + testCorrect + "," + trainCorrect + "\n");
				resultCsvWriter.flush();
//				iter.remove();
			}			
		}
		// write results on screen
		for(Map.Entry<Double, String> entry : sortedResults.entrySet())
		{
			System.out.print("BoW_features: " + entry.getValue() + "\t");
			System.out.println("correctly classified: " + entry.getKey());
		}
		resultCsvWriter.close();
	}
}
