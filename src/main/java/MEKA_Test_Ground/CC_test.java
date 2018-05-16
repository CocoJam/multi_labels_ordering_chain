package MEKA_Test_Ground;

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import scala.Enumeration;
import scala.math.Numeric;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.core.pmml.jaxbbindings.SupportVectorMachine;
import weka.filters.unsupervised.attribute.NumericToBinary;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CC_test {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
        Instances data2 = source.getDataSet();
        Instances data = new Instances(data2);
        double splitRate = 66;
        int trainSize = (int) (data.numInstances() * splitRate / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.println("Build CC classifier");

        CC cc = new CC();
        MLUtils.prepareData(data2);
        cc.buildClassifier(data2);
    }
}
