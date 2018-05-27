package WEKA_Test_Ground;

import Prepare_CC.Base_CC;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.MLUtils;
import meka.core.Result;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class BaggingMLEvaluation {
    public static void main(String[] args) throws Exception {
        Instances train = (new ConverterUtils.DataSource("src/main/CAL500.arff")).getDataSet();
//        Instances train = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_train.arff")).getDataSet();

        train.deleteAttributeAt(train.numAttributes()-1);
//        Instances test = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_test.arff")).getDataSet();
//        Base_CC cc = new Base_CC();
        BaggingML baggingML = new BaggingML();
        MLUtils.prepareData(train);
//        MLUtils.prepareData(test);
        String top = "PCut1";
        String vop = "3";
        baggingML.buildClassifier(train);
        Result evaluateModel;
        evaluateModel = Evaluation.cvModel(baggingML, train, 10 ,top,vop);
        System.out.println(evaluateModel);

    }
}
