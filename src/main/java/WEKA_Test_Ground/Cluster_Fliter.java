package WEKA_Test_Ground;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class Cluster_Fliter {
    public static Instances importArff(String file){
        ConverterUtils.DataSource source = null;
        try {
            source = new ConverterUtils.DataSource("src/main/CAL500.arff");
            return source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    };

    public static Instances filter(int value, Attribute attribute){
        return null;
    }

    public static void main(String[] args) {
        ConverterUtils.DataSource source = null;
        Instances data =null;
        try {
            source = new ConverterUtils.DataSource("src/main/CAL500_clustered.arff");
            data = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
        }
//        Attribute a = new Attribute("Cluster");
        RemoveWithValues filter = new RemoveWithValues();
        filter.setAttributeIndex("1");
        try {
            filter.setSplitPoint(2);
//            System.out.println(data);
//            filter.setInvertSelection(true);
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            System.out.println(newData);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
