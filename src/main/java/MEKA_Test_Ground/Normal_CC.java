package MEKA_Test_Ground;

import meka.classifiers.multilabel.*;
import meka.core.MLUtils;
import meka.core.Result;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Normal_CC extends ProblemTransformationMethod {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = null;
        double splitRate = 0.66;
        try {

            reader = new BufferedReader(
                    new FileReader("src/main/mediamill.arff"));

            Instances data = new Instances(reader);

            MLUtils.prepareData(data);
            int trainSize = (int) (data.numInstances() * splitRate / 100.0);
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
            System.out.println("Build CC classifier");
            //This CC class is using weka.classifiers.trees.j48.C45PruneableClassifierTree such that it is not made to
            //handle numeric class hence the Mediamill.arff is not build. Might need to use Mulan
            PCC classifier = new PCC();

            classifier.buildClassifier(data);
            System.out.println("Evaluate BR classifier on " + (100.0 - splitRate) + "%");
            String top = "PCut1";
            String vop = "3";
            Result result = Evaluation.evaluateModel(classifier, train, test, top, vop);
            // further configuration of classifier

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            reader.close();
        }
        //This is the method to read the file and parse everything in to instances

    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        testCapabilities(instances);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int C = instance.classIndex();
        return new double[C];
    }
}
