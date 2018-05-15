package MEKA_Test_Ground;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CC_test {

    public static void ccRun(int clusterNum, int splitRate, String file, int[] ints) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();
        data = Cluster_Fliter.filter(data,clusterNum);
        int trainSize = (int) (data.numInstances() * splitRate / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);
        System.out.println("Build CC classifier");
        System.out.println(data.relationName());
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        long time1= System.nanoTime();
        CC cc = new CC();
        int[] ar = new int[numLabels];
        for (int i = 0; i < numLabels; i++) {
            ar[i] = i;
        }
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
        ints = ints == null? ar: ints;
        System.out.println(Arrays.toString(ar));
        cc.prepareChain(ints);
        MLUtils.prepareData(data);
        cc.buildClassifier(data);
        String top = "PCut1";
        String vop = "3";
        int numOfCV = data.numInstances()>10? 10:data.numInstances();
        Result result = Evaluation.cvModel(cc, data, numOfCV, top, vop);
        System.out.println(result);
        System.out.println(Arrays.toString(cc.retrieveChain()));
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
    }
    public static void main(String[] args) throws Exception {
        long time1= System.nanoTime();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
        for (int i = 0; i < 8; i++) {
            CC_test.ccRun(i,66,"src/main/CAL500_clustered_adjusted.arff",null);
        }
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
    }
}
