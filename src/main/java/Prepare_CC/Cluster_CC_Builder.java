package Prepare_CC;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.MLUtils;
import meka.core.Result;
import scala.Int;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.lang.reflect.Array;
import java.util.*;
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
    public int parsedLabelNum;
    public int overallLabelNum;

    public Cluster_CC_Builder(String dataSource, int clusterNum, double threadshold) throws Exception {
        this.clusterNum = clusterNum;
        this.dataSource = dataSource;
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataSource);
        Instances data = source.getDataSet();
        if (clusterNum != -1) {
            this.cluster = Cluster_Fliter.filter(data, clusterNum);
        } else {
            this.cluster = data;
        }
        this.parsedCluster = new Instances(this.cluster);

        setUp(this.parsedCluster, threadshold,true);
    }

    public Cluster_CC_Builder(int clusterNum, Instances data, double threadshold) throws Exception {
        this.clusterNum = clusterNum;
        this.cluster = Cluster_Fliter.filter(data, clusterNum);
        this.parsedCluster = new Instances(this.cluster);
        setUp(this.parsedCluster, threadshold, true);
    }

    public Cluster_CC_Builder(Instances data, double threadshold) throws Exception {
        this.parsedCluster = new Instances(data);
        this.cluster = this.parsedCluster;
        CC cc = new CC();
        MLUtils.prepareData(data);
        cc.buildClassifier(data);
        setUp(this.parsedCluster, threadshold, false);
    }

    private void setUp(Instances data, double threadshold, boolean clustered) throws Exception {
        Pattern pattern = Pattern.compile("((.+-C )(\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        String group2 = "";
        if (matcher.find()) {
            data.setRelationName(matcher.group(0));
            group2 = matcher.group(2);
            numLabels = Integer.parseInt(matcher.group(3));
        }
        overallLabelNum = numLabels;
        int[] listList = new int[numLabels];

        double[] featureList = new double[data.numAttributes() - numLabels - 1];
        for (int j = 0; j < cluster.numInstances(); j++) {
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                if (i < (numLabels)) {
                    listList[i] -= (int) cluster.get(j).value(i);
                } else {
                    featureList[i - numLabels] += cluster.get(j).value(i) / data.numInstances();
                }
            }
        }
        this.featureVector = featureList;
        List<Integer> ListOfInt = new ArrayList<>();
        double degrees = (cluster.numInstances() * threadshold) * -1;
        int missingLabelCount = 0;
        if (clustered) {
            for (int i = listList.length - 1; i >= 0; i--) {
                if (listList[i] < degrees) {
                    ListOfInt.add(i);
                } else {
                    this.parsedCluster.deleteAttributeAt(i);
                    missingLabelCount++;
                }
            }
            Collections.sort(ListOfInt);
            int[] blah = Arrays.stream(listList).map(p -> {
                if (p == 0) {
                    return p;
                }
                return 1;
            }).toArray();
            data.setRelationName(group2 + (numLabels - (missingLabelCount)));
            this.labelChain = Arrays.stream(ListOfInt.toArray(new Integer[ListOfInt.size()])).mapToInt(Integer::intValue).toArray();
            parsedLabelNum = this.labelChain.length;
            this.sqeuenceChain = new int[this.labelChain.length];
            for (int i = 0; i < this.labelChain.length; i++) {
                this.sqeuenceChain[i] = i;
            }
        } else {
            int[] labelChain = new int[numLabels];
            for (int i = 0; i < numLabels; i++) {
                labelChain[i]=i;
            }
            this.parsedCluster = data;
            this.labelChain = labelChain;
            this.parsedLabelNum = labelChain.length;
            this.sqeuenceChain =labelChain;
        }
    }

    public static void main(String[] args) throws Exception {
        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff", 3, 0);
        System.out.println(cluster_cc_builder.labelChain.length);

        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500.arff");
        Instances data = source.getDataSet();
        BaggingML baggingML = new BaggingML();
        CC base_cc = new CC();
//        base_cc.prepareChain(cluster_cc_builder.sqeuenceChain);
//        baggingML.setClassifier();
        MLUtils.prepareData(data);

        baggingML.buildClassifier(data);

        String top = "PCut1";
        String vop = "3";
        int numOfCV = data.numInstances() > 10 ? 10 : data.numInstances();
        Result result = Evaluation.cvModel(baggingML, data, numOfCV, top, vop);
        double hamming_loss = Double.parseDouble(result.getMeasurement("Hamming score").toString());
        double exact_match = Double.parseDouble(result.getMeasurement("Exact match").toString());
        double accuracy = Double.parseDouble(result.getMeasurement("Accuracy").toString());
        double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;
        System.out.println(result);
        System.out.println(averaging);

    }
}
