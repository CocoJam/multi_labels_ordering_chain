package WEKA_Test_Ground;

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Knn_Inferencing {
    public static void main(String[] args) throws Exception {

        Instances data = (new ConverterUtils.DataSource("src/main/CAL500_test.arff")).getDataSet();
        Instances dataTest = new Instances(data);
        int trainSize = (int) (data.numInstances() * 66.0 / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

//        System.out.println(dataTest);
        MLUtils.prepareData(train);
        MLUtils.prepareData(test);
        CC cc = new CC();
        cc.buildClassifier(train);
        String top = "PCut1";
        String vop = "3";
        Result e = Evaluation.evaluateModel(cc,train,test,top,vop);
//        double hamming_score = e.getMeasurement("Hamming score");
        System.out.println(e);

    }
}
