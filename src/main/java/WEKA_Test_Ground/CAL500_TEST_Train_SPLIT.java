package WEKA_Test_Ground;

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
        try {
            source = new ConverterUtils.DataSource("src/main/CAL500.arff");
            Instances data = source.getDataSet();
//            Random random = new Random();
//            random.setSeed(seed);
            int trainSize = (int) (data.numInstances() * 66.0 / 100.0);
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
            ArffSaver saver = new ArffSaver();
//            System.out.println(newData);
            saver.setInstances(train);
            saver.setFile(new File("src/main/CAL500_train.arff"));
            saver.writeBatch();
            ArffSaver saver1 = new ArffSaver();
            saver1.setInstances(test);
            saver1.setFile(new File("src/main/CAL500_test.arff"));
            saver1.writeBatch();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
