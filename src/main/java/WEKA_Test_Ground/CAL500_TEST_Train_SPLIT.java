package WEKA_Test_Ground;

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

    public static void main(String[] args) {
        ConverterUtils.DataSource source = null;
        for (int j = 0; j < 10; j++) {
            try {
                File file = new File("Split_" + j);
                if (!file.exists()) {
                    file.mkdir();
                }

                source = new ConverterUtils.DataSource("src/main/CAL500.arff");
                Instances data = source.getDataSet();
                Random random = new Random();
                random.setSeed(System.currentTimeMillis());
                data.randomize(random);
                int trainSize = (int) (data.numInstances() * 66.0 / 100.0);
                Instances train = new Instances(data, 0, trainSize);
                Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
                EM EM_test = new EM();
                //Seeding for eval
                EM_test.setSeed(10);
                EM_test.buildClusterer(train);
                //eval object for any clusters
                ClusterEvaluation eval = new ClusterEvaluation();
                eval.setClusterer(EM_test);
                eval.evaluateClusterer(train);
                Instances newData = new Instances(train);
                Attribute attribute = new Attribute("Cluster");
                newData.insertAttributeAt(attribute,newData.numAttributes());
                for (int i = 0; i < newData.numInstances(); i++) {
                    System.out.println(newData.get(i).attribute(data.get(i).numAttributes()));
//                System.out.println(EM_test.clusterInstance(newData.get(i)));
                    newData.get(i).setValue(newData.get(i).attribute(newData.get(i).numAttributes()-1), EM_test.clusterInstance(data.get(i)));
                }
                ArffSaver saver = new ArffSaver();
                System.out.println(newData);
                saver.setInstances(newData);
//            System.out.println(newData);
//                saver.setInstances(train);
                saver.setFile(new File("Split_"+j+"/CAL500_train.arff"));
                saver.writeBatch();
                ArffSaver saver1 = new ArffSaver();
                saver1.setInstances(test);
                saver1.setFile(new File("Split_"+j+"/CAL500_test.arff"));
                saver1.writeBatch();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }
}
