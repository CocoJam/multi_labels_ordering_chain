package Prepare_CC;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.instance.SubsetByExpression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CC_Util {

    public static Instances filter(Instances source,int value){
        SubsetByExpression subsetByExpression =new SubsetByExpression();
        subsetByExpression.setExpression("ATT"+(source.attribute("Cluster").index()+1)+"="+value);
        try {
            subsetByExpression.setInputFormat(source);
            return SubsetByExpression.useFilter(source, subsetByExpression);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void ccRun(String file, int clusterNum, int splitRate, int[] ints) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();
        data = CC_Util.filter(data,clusterNum);
        ccRunAndBuildAndEval(splitRate,data, ints);
    }

    public static double ccRun(Cluster_CC_Builder cluster_cc_builder, int splitRate) throws Exception {
        Instances data = cluster_cc_builder.parsedCluster;
//        System.out.println(data);
        return ccRunAndBuildAndEval(splitRate, cluster_cc_builder.parsedCluster ,cluster_cc_builder.sqeuenceChain);
    }

    public static double ccRun(Cluster_CC_Builder cluster_cc_builder, int splitRate, int[] labelChainPrepared) throws Exception {
        Instances data = cluster_cc_builder.parsedCluster;
//        System.out.println(data);
        return ccRunAndBuildAndEval(splitRate, cluster_cc_builder.parsedCluster ,labelChainPrepared);
    }

//    private static void ccRunAndBuildAndEval(Cluster_CC_Builder cluster_cc_builder) throws Exception {
//        Instances data = cluster_cc_builder.parsedCluster;
//        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
//        Matcher matcher = pattern.matcher(data.relationName());
//        int numLabels = 0;
//        if(matcher.find()){
//            data.setRelationName(matcher.group(0));
//            numLabels = Integer.parseInt(matcher.group(2));
//        }
//        long time1= System.nanoTime();
//
//        int[] ar = new int[numLabels];
//        for (int i = 0; i < numLabels; i++) {
//            ar[i] = i;
//        }
//        Random rnd = ThreadLocalRandom.current();
//        for (int i = ar.length - 1; i > 0; i--)
//        {
//            int index = rnd.nextInt(i + 1);
//            int a = ar[index];
//            ar[index] = ar[i];
//            ar[i] = a;
//        }
//        CC cc = new CC();
//        MLUtils.prepareData(data);
//        cc.buildClassifier(data);
//        cc.rebuildClassifier(cluster_cc_builder.labelChain,cluster_cc_builder.parsedCluster);
//        String top = "PCut1";
//        String vop = "3";
//        int numOfCV = data.numInstances()>10? 10:data.numInstances();
//        Result result = Evaluation.cvModel(cc, data, numOfCV, top, vop);
//        System.out.println(result);
//        System.out.println(Arrays.toString(cc.retrieveChain()));
//        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
//        System.out.println(time2);
//        result.getInfo("");
//    }

    private static double ccRunAndBuildAndEval(int splitRate, Instances data, int[] ints) throws Exception {
        int trainSize = (int) (data.numInstances() * splitRate / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        long time1= System.nanoTime();

        Base_CC cc = new Base_CC();
        cc.prepareChain(ints);
        MLUtils.prepareData(data);
        System.out.println("Building");
        cc.buildClassifier(data);

//        cc.rebuildClassifier(ints,data);
        String top = "PCut1";
        String vop = "3";
        int numOfCV = data.numInstances()>10? 10:data.numInstances();
        Result result = Evaluation.cvModel(cc, data, numOfCV, top, vop);
//        System.out.println(result);
//        System.out.println(Arrays.toString(cc.retrieveChain()));
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
//        for (String s : result.availableMetrics()) {
//            System.out.print(s+": ");
//            System.out.println(result.getMeasurement(s));
//        }
//        System.out.println(result.getMeasurement("Hamming score"));
//        System.out.println(result.getMeasurement("Exact match"));
//        System.out.println(result.getMeasurement("Accuracy"));
        double hamming_loss= Double.parseDouble(result.getMeasurement("Hamming score").toString());
        double exact_match= Double.parseDouble(result.getMeasurement("Exact match").toString());
        double accuracy= Double.parseDouble(result.getMeasurement("Accuracy").toString());
        double averaging = ((1-hamming_loss)+ exact_match +accuracy)/3;
        return averaging;
    }

    public static List<Integer[]> labelOrderChains(String file, int clusterAmount) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();
        return getChainsList(clusterAmount, data);
    }

    private static List<Integer[]> getChainsList(int clusterAmount, Instances data) {
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }

        List<Integer[]> listOfLabels= new ArrayList<>();
        List<Integer[]> listOfNonLabels = new ArrayList<>();
        for (int k = 0; k < clusterAmount; k++) {
            Instances dataFliter = Cluster_Fliter.filter(data,k);
            int[] listList = new int[numLabels];
            for (int j = 0; j < dataFliter.numInstances(); j++) {
                for (int i = 0; i < numLabels; i++) {
                    listList[i] -= (int) dataFliter.get(j).value(i);
                }
            }
            List<Integer> ListOfInt = new ArrayList<>();
            for (int i = 0; i < listList.length; i++) {
                if(listList[i]<0){
                    ListOfInt.add(i);
                }
            }
            Integer[] intArrayLabels = new Integer[ListOfInt.size()];
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            listOfLabels.add(intArrayLabels);
        }
        return listOfLabels;
    }

    public static void main(String[] args) throws Exception {
//        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
//        Instances data = source.getDataSet();
//        List<Integer[]> integers  = CC_Util.getChainsList(8,data);
//        for (int i = 0; i < 8; i++) {
//            System.out.println(Arrays.toString(integers.get(i)));
            Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff",0,0);
//            CC_Util.ccRun(cluster_cc_builder,66);
//            System.out.println(Arrays.toString(cluster_cc_builder.labelChain));
//        }
    }
}
