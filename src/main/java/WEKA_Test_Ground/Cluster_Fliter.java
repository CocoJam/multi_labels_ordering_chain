package WEKA_Test_Ground;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.filters.unsupervised.instance.SubsetByExpression;

public class Cluster_Fliter {

    public static Instances filter(Instances source,int value){
        SubsetByExpression subsetByExpression =new SubsetByExpression();
        subsetByExpression.setExpression("ATT1="+value);
        try {
            subsetByExpression.setInputFormat(source);
            return SubsetByExpression.useFilter(source, subsetByExpression);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered.arff");
        Instances data = source.getDataSet();
        System.out.println( Cluster_Fliter.filter(data,2));
    }
}
