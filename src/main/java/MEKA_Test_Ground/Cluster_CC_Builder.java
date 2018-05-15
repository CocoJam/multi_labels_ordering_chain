package MEKA_Test_Ground;

import WEKA_Test_Ground.Cluster_Fliter;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Cluster_CC_Builder {
public Integer[] labelChain;
public Instances cluster;
public Instances parsedCluster;
public int clusterNum;
public String dataSource;

    public Cluster_CC_Builder(int clusterNum, String dataSource) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataSource);
        Instances data = source.getDataSet();
        this.cluster=Cluster_Fliter.filter(data, clusterNum);
        this.parsedCluster = new Instances(this.cluster);
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        int[] listList = new int[numLabels];
        for (int j = 0; j < cluster.numInstances(); j++) {
            for (int i = 0; i < numLabels; i++) {
                listList[i] -= (int) cluster.get(j).value(i);
            }
        }
        List<Integer> ListOfInt = new ArrayList<>();
        List<Integer> ListOfNonInt = new ArrayList<>();
        for (int i = 0; i < listList.length; i++) {
//                List<Integer> ListOfInt = new ArrayList<>();
            if(listList[i]<0){
                ListOfInt.add(i);
            }else{
                this.parsedCluster.deleteAttributeAt(i);
            }
        }
    }
}
