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
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_blah.arff");
        Instances data2 = source.getDataSet();
        Instances data = new Instances(data2);
//        java.util.Enumeration<Attribute> asd= data.enumerateAttributes();
//        List<Attribute> ListLabel = new ArrayList<>();
//        List<Attribute> ListAttr = new ArrayList<>();
//        while(asd.hasMoreElements()){
//           Attribute attribute =asd.nextElement();
//           if ( attribute.type()!=0){
//               ListLabel.add(attribute);
//           }
//           else{
//               ListAttr.add(attribute);
//           }
//        }
//        java.util.Enumeration<Instance> instances = data.enumerateInstances();
//        while(instances.hasMoreElements()){
//            instances.nextElement().
//        }

////        System.out.println(i);
//        int[] array = Listint.stream().mapToInt(j->j).toArray();
//        NumericToBinary numericToBinary =new NumericToBinary();
//        numericToBinary.setAttributeIndicesArray(array);
//        numericToBinary.setInputFormat(data);
//        Instances blah =NumericToBinary.useFilter(data,numericToBinary);
//
//////        System.out.println(data);
//        ArffSaver saver = new ArffSaver();
//        saver.setInstances(data);
//        saver.setFile(new File("src/main/CAL500_clustered_blah.arff"));
//        System.out.println(blah);
        double splitRate = 66;
        int trainSize = (int) (data.numInstances() * splitRate / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.println("Build CC classifier");
        CC cc = new CC();
        SMO classifier = new SMO();
//        System.out.println(classifier.getCapabilities());
////        classifier.setOptions(new String[] { "-R" });
        MLUtils.prepareData(data2);
//        System.out.println(data2);
//        cc.setClassifier(classifier);
        cc.buildClassifier(data2);
//        String top = "PCut1";
//        String vop = "3";
//        Result result = Evaluation.evaluateModel(cc, train, test, top, vop);
//        System.out.println(result);
    }
}
