package WEKA_Test_Ground;

import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.MLUtils;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.Random;

public class CAL500_TEST_Train_SPLIT {

    public CAL500_TEST_Train_SPLIT(String name, double split, int seed) {
        ConverterUtils.DataSource source = null;
        try {
            source = new ConverterUtils.DataSource(name);
            Instances data = source.getDataSet();
//            Random random = new Random();
//            random.setSeed(seed);
            int trainSize = (int) (data.numInstances() * split / 100.0);
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
            ArffSaver saver = new ArffSaver();
//            System.out.println(newData);
            saver.setInstances(train);
            saver.setFile(new File("src/main/CAL500_train.arff"));
            saver.writeBatch();
            ArffSaver saver1 = new ArffSaver();
            saver.setInstances(test);
            saver1.setFile(new File("src/main/CAL500_test.arff"));
            saver1.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) throws Exception {
//                ConverterUtils.DataSource source = null;
//        source = new ConverterUtils.DataSource("src/main/CAL500.arff");
//        Instances data = source.getDataSet();
//        MLUtils.prepareData(data);
//        BaggingML baggingML = new BaggingML();
//        baggingML.buildClassifier(data);
//        String top = "PCut1";
//        String vop = "3";
//        meka.core.Result result = Evaluation.cvModel(baggingML,data,10,top,vop);
//        System.out.println(result);


//        ConverterUtils.DataSource source = null;
//        source = new ConverterUtils.DataSource("src/main/CAL500.arff");
//        Instances data = source.getDataSet();
//        Random random = new Random();
//        data.randomize(random);
//        System.out.println(data.instance(0));
//        data.randomize(random);
//        System.out.println(data.instance(0));

        ConverterUtils.DataSource source = null;
        source = new ConverterUtils.DataSource("src/main/CAL500.arff");
        Instances data = source.getDataSet();
        Random random = new Random();
        for (int i = 0; i < 10; i++) {
            random.setSeed(i);
            data.randomize(random);
            for (int j = 0; j < 10; j++) {
                try {
                    File file = new File("CVseed_"+i);
                    if (!file.exists()) {
                        file.mkdir();
                    }
                    File file1 = new File("CVseed_"+i+"/"+"CVSplit_" + j);
                    if (!file1.exists()) {
                        file1.mkdir();
                    }
                    Instances train = data.trainCV(10, j);
                    Instances test = data.testCV(10, j);

                    EM EM_test = new EM();
                    //Seeding for eval
                    EM_test.setSeed(i);
                    EM_test.buildClusterer(train);
                    //eval object for any clusters
                    ClusterEvaluation eval = new ClusterEvaluation();
                    eval.setClusterer(EM_test);
                    eval.evaluateClusterer(train);
                    Instances newData = new Instances(train);
                    Attribute attribute = new Attribute("Cluster");
                    newData.insertAttributeAt(attribute,newData.numAttributes());
                    for (int w = 0; w < newData.numInstances(); w++) {
//                    System.out.println(newData.get(i).attribute(data.get(i).numAttributes()));
//                System.out.println(EM_test.clusterInstance(newData.get(i)));
                        newData.get(w).setValue(newData.get(w).attribute(newData.get(w).numAttributes()-1), EM_test.clusterInstance(data.get(w)));
                    }
                    ArffSaver saver = new ArffSaver();
                    System.out.println(newData);
                    saver.setInstances(newData);
//            System.out.println(newData);
//                saver.setInstances(train);
                    saver.setFile(new File("CVseed_"+i+"/"+"CVSplit_" + j+"/CAL500_train.arff"));
                    saver.writeBatch();
                    ArffSaver saver1 = new ArffSaver();
                    saver1.setInstances(test);
                    saver1.setFile(new File("CVseed_"+i+"/"+"CVSplit_" + j+"/CAL500_test.arff"));
                    saver1.writeBatch();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }


    }
}
