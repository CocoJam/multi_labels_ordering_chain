package Prepare_CC;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.core.MLUtils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Cluster_CC_Builder {
    public int[] labelChain;
    public int[] sqeuenceChain;
    public Instances cluster;
    public Instances parsedCluster;
    public int clusterNum;
    public String dataSource;
    public double[] featureVector;

    public Cluster_CC_Builder(String dataSource, int clusterNum, double threadshold) throws Exception {
        this.clusterNum = clusterNum;
        this.dataSource = dataSource;
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataSource);
        Instances data = source.getDataSet();
        this.cluster = Cluster_Fliter.filter(data, clusterNum);
        this.parsedCluster = new Instances(this.cluster);

        setUp(this.parsedCluster, threadshold);
    }

    public Cluster_CC_Builder(int clusterNum, Instances data, double threadshold) throws Exception {
        this.clusterNum = clusterNum;
        this.cluster = Cluster_Fliter.filter(data, clusterNum);
        this.parsedCluster = new Instances(this.cluster);
        setUp(this.parsedCluster, threadshold);
    }

    public Cluster_CC_Builder(Instances data, double threadshold) throws Exception {
        this.parsedCluster = new Instances(data);
        CC cc = new CC();
        MLUtils.prepareData(data);
        cc.buildClassifier(data);
        System.out.println("building");
        setUp(this.parsedCluster, threadshold);
    }

    private void setUp(Instances data, double threadshold) throws Exception {
        Pattern pattern = Pattern.compile("((.+-C )(\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        String group2 = "";
        if (matcher.find()) {
            data.setRelationName(matcher.group(0));
            group2 = matcher.group(2);
            numLabels = Integer.parseInt(matcher.group(3));
        }
        int[] listList = new int[numLabels];
        double[] featureList = new double[data.numAttributes() - numLabels-1];
        for (int j = 0; j < cluster.numInstances(); j++) {
            for (int i = 0; i < data.numAttributes()-1; i++) {
                if (i < numLabels) {
                    listList[i] -= (int) cluster.get(j).value(i);
                } else {
                    featureList[i-numLabels] += cluster.get(j).value(i)/data.numInstances();
                }
            }
        }
        this.featureVector = featureList;
        List<Integer> ListOfInt = new ArrayList<>();
        double degrees = cluster.numInstances() * threadshold;
        int missingLabelCount = 0;
        for (int i = 0; i < listList.length; i++) {
            if (listList[i] < degrees) {
                ListOfInt.add(i);
            } else {
                this.parsedCluster.deleteAttributeAt(i);
                missingLabelCount++;
            }
        }
        data.setRelationName(group2 + (numLabels - (missingLabelCount)));
        //Bug found mis match in the parsed Cluster with the orginal will try to fix it.
//        CC cc = new CC();
//
//        MLUtils.prepareData(data);
//        cc.buildClassifier(data);
        System.out.println("building");
        System.out.println(data.relationName());
        this.labelChain = Arrays.stream(ListOfInt.toArray(new Integer[ListOfInt.size()])).mapToInt(Integer::intValue).toArray();
        this.sqeuenceChain = new int[this.labelChain.length];
        for (int i = 0; i < this.labelChain.length; i++) {
            this.sqeuenceChain[i] = i;
        }
        System.out.println(Arrays.toString(this.labelChain));
    }


}
