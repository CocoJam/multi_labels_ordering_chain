package WEKA_Test_Ground;

import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Knn_Inferencing {
    public static void main(String[] args) throws Exception {

        Instances dataClus = (new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff")).getDataSet();
        Instances dataCluster = new Instances(dataClus);
        Remove remove = new Remove();
        int[] blah = new int[174];
        for (int i = 0; i < blah.length; i++) {
            blah[i] = i;
        }
        remove.setAttributeIndicesArray(blah);
        remove.setInputFormat(dataClus);
        dataClus  = Filter.useFilter(dataClus, remove);

        IBk iBk = new IBk();
        Instances data = (new ConverterUtils.DataSource("src/main/CAL500_test.arff")).getDataSet();
        Instances dataTest = new Instances(data);
        dataClus.setClassIndex(dataClus.numAttributes()-1);
        iBk.buildClassifier(dataClus);
        Attribute attribute = new Attribute("Cluster");

        data.insertAttributeAt(attribute,data.numAttributes());
        data  = Filter.useFilter(data, remove);
        data.setClassIndex(data.numAttributes()-1);

        iBk.setKNN(5);
        for (int i = 0; i < dataTest.numInstances(); i++) {
            double d =iBk.classifyInstance(data.get(i));
            System.out.println(Math.round(d));
            dataTest.get(i).setValue(dataTest.get(i).attribute(dataTest.get(i).numAttributes()-1), Math.round(d));
        }

        System.out.println(dataTest);


    }
}
