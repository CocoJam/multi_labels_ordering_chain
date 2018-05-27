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

    public static void main(String[] args) throws Exception {
//        ConverterUtils.DataSource source = null;
//        source = new ConverterUtils.DataSource("src/main/CAL500.arff");
//        Instances data = source.getDataSet();
//
//        for (int j = 0; j < 10; j++) {
//            try {
//
//                File file = new File("CVSplit_" + j);
//                if (!file.exists()) {
//                    file.mkdir();
//                }
//                Instances train = data.trainCV(10, j);
//                Instances test = data.testCV(10, j);
//
//                EM EM_test = new EM();
//                //Seeding for eval
//                EM_test.setSeed(10);
//                EM_test.buildClusterer(train);
//                //eval object for any clusters
//                ClusterEvaluation eval = new ClusterEvaluation();
//                eval.setClusterer(EM_test);
//                eval.evaluateClusterer(train);
//                Instances newData = new Instances(train);
//                Attribute attribute = new Attribute("Cluster");
//                newData.insertAttributeAt(attribute,newData.numAttributes());
//                for (int i = 0; i < newData.numInstances(); i++) {
////                    System.out.println(newData.get(i).attribute(data.get(i).numAttributes()));
////                System.out.println(EM_test.clusterInstance(newData.get(i)));
//                    newData.get(i).setValue(newData.get(i).attribute(newData.get(i).numAttributes()-1), EM_test.clusterInstance(data.get(i)));
//                }
//                ArffSaver saver = new ArffSaver();
//                System.out.println(newData);
//                saver.setInstances(newData);
////            System.out.println(newData);
////                saver.setInstances(train);
//                saver.setFile(new File("CVSplit_"+j+"/CAL500_train.arff"));
//                saver.writeBatch();
//                ArffSaver saver1 = new ArffSaver();
//                saver1.setInstances(test);
//                saver1.setFile(new File("CVSplit_"+j+"/CAL500_test.arff"));
//                saver1.writeBatch();
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }
        String blah = "[31, 156, 30, 39, 166, 55, 7, 25, 141, 72, 89, 84, 140, 3, 172, 138, 144, 41, 112, 155, 160, 158, 145, 54, 98, 23, 15, 69, 137, 94, 60, 53, 49, 164, 162, 95, 171, 90, 152, 99, 135, 129, 136, 32, 110, 159, 105, 75, 104, 26, 40, 100, 123, 151, 148, 16, 56, 74, 109, 65, 73, 51, 126, 10, 161, 79, 114, 157, 83, 28, 165, 19, 44, 12, 142, 64, 52, 82, 169, 101, 14, 38, 150, 63, 117, 35, 70, 102, 2, 118, 111, 20, 116, 57, 163, 88, 91, 34, 42, 168, 4, 45, 86, 139, 59, 29, 124, 47, 46, 17, 113, 8, 154, 125, 66, 9, 134, 143, 106, 76, 173, 50, 27, 146, 149, 22, 24, 6, 77, 1, 120, 0, 93, 115, 5, 96, 131, 71, 48, 132, 78, 107, 170, 61, 85, 68, 21, 18, 127, 108, 43, 121, 33, 36, 58, 13, 128, 67, 62, 167, 103, 11, 80, 122, 97, 119, 92, 130, 133, 37, 147, 87, 153, 81]\n";
        System.out.println(blah.split(",").length);
    }
}
