package WEKA_Test_Ground;

import Prepare_CC.Cluster_CC_Builder;
import org.ejml.simple.SimpleMatrix;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.SubsetByExpression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Cluster_Fliter {

    public static Instances filter(Instances source,int value){
        SubsetByExpression subsetByExpression =new SubsetByExpression();
        subsetByExpression.setExpression("ATT"+(source.attribute("Cluster").index()+1)+"="+value);
        try {
            subsetByExpression.setInputFormat(source);
           source= SubsetByExpression.useFilter(source, subsetByExpression);
            Pattern pattern = Pattern.compile("(.+-C (\\d+))");
            Matcher matcher = pattern.matcher(source.relationName());
            if (matcher.find()) {
                source.setRelationName(matcher.group(0));
            }
            return source;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Instances knn_inference(String trainingSet, String testSet, int k, int labelnum) throws Exception {
        Instances dataClus = (new ConverterUtils.DataSource(trainingSet)).getDataSet();
        Instances train = new Instances(dataClus);
        Instances data = (new ConverterUtils.DataSource(testSet)).getDataSet();
        Instances test = new Instances(data);

        return  knn_inference(train, test,k);
    }


    public static Instances knn_inference(Instances trainingSet, Instances testSet, int k) throws Exception {
        Instances dataCluster = new Instances(trainingSet);
//        System.out.println(dataCluster.numAttributes());
        Instances dataTest = new Instances(testSet);
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(dataTest.relationName());
        int numLabels = 0;
        if (matcher.find()) {
            dataTest.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        int[] blah = new int[numLabels];
        for (int i = 0; i < blah.length; i++) {
            blah[i] = i;
        }
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(blah);
        remove.setInputFormat(trainingSet);
        trainingSet = Filter.useFilter(trainingSet, remove);
        IBk iBk = new IBk();
        trainingSet.setClassIndex(trainingSet.numAttributes()-1);
        iBk.buildClassifier(trainingSet);
        Attribute attribute = new Attribute("Cluster");

        testSet.insertAttributeAt(attribute,testSet.numAttributes());
        remove.setInputFormat(testSet);
        testSet  = Filter.useFilter(testSet, remove);
        testSet.setClassIndex(testSet.numAttributes()-1);
//        System.out.println(testSet.numAttributes());
//        System.out.println(trainingSet.numAttributes());
        iBk.setKNN(k);
        dataTest.insertAttributeAt(attribute,dataTest.numAttributes());
        for (int i = 0; i < dataTest.numInstances(); i++) {
            double d =iBk.classifyInstance(testSet.get(i));
//            System.out.println(Math.round(d));
            dataTest.get(i).setValue(dataTest.get(i).attribute(dataTest.get(i).numAttributes()-1), Math.round(d));
        }
//        System.out.println(dataTest.numAttributes());
        return dataTest;
    }

}
