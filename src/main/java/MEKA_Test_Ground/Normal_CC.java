package MEKA_Test_Ground;

import meka.classifiers.multilabel.*;
import meka.core.MLUtils;
import meka.core.Result;

import weka.core.Instance;
import weka.core.Instances;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Timer;

public class Normal_CC extends ProblemTransformationMethod {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = null;
        double splitRate = 66;
        try {

            reader = new BufferedReader(
                    new FileReader("src/main/CAL500.arff"));

            Instances data = new Instances(reader);
//            NumericToNominal convert= new NumericToNominal();
//            String[] options= new String[2];
//            options[0]="-R";
//            options[1]="0-1";  //range of variables to make numeric
//            convert.setOptions(options);
//            convert.setInputFormat(data);

//            data= Filter.useFilter(data, convert);
            CC cc = new CC();

            String[] ccop = cc.getOptions();
            ccop[3] = "weka.classifiers.trees.REPTree";
//            ccop[3] = "weka.classifiers.functions.MultilayerPerceptron";
            String[] newOp = new String[5];
            for (int i = 0; i < newOp.length; i++) {
                newOp[i] = ccop[i];
            }
            for (String s : newOp) {
                System.out.println(s);
            }
            cc.setOptions(newOp);

            MLUtils.prepareData(data);
            int trainSize = (int) (data.numInstances() * splitRate / 100.0);
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
            System.out.println("Build CC classifier");
            //This CC class is using weka.classifiers.trees.j48.C45PruneableClassifierTree such that it is not made to
            //handle numeric class hence the Mediamill.arff is not build. Might need to use Mulan

            long startTime = System.currentTimeMillis();

            cc.buildClassifier(data);

            long endTime = System.currentTimeMillis();

            System.out.println("That took " + (endTime - startTime) + " milliseconds");

            System.out.println("hello");
            System.out.println("Evaluate BR classifier on " + (100.0 - splitRate) + "%");
            String top = "PCut1";
            String vop = "3";
            Result result = Evaluation.evaluateModel(cc, train, test, top, vop);
            System.out.println(result);
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
