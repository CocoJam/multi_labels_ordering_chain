package MEKA_Test_Ground;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.core.MLUtils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.concurrent.TimeUnit;

public class CC_test {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
        Instances data = source.getDataSet();
        data = Cluster_Fliter.filter(data,7);
//        System.out.println(data.instance(0).attribute(i-1));
        double splitRate = 66;
        int trainSize = (int) (data.numInstances() * splitRate / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.println("Build CC classifier");
        long time1= System.nanoTime();
        CC cc = new CC();
        MLUtils.prepareData(data);
        cc.buildClassifier(data);
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
    }
}
