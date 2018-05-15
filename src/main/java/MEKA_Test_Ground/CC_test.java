package MEKA_Test_Ground;

import Prepare_CC.CC_Util;
import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class CC_test {

    public static void ccRun(int clusterNum, int splitRate, String file, int[] ints) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();
        data = CC_Util.filter(data,clusterNum);
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
//        ints = ints == null? ar: ints;
//        System.out.println(Arrays.toString(ar));
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

    public List<Integer[]> labelOrderChains(String file, int clusterNum) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();

        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }

        List<Integer[]> listOfLabels= new ArrayList<>();
        List<Integer[]> listOfNonLabels = new ArrayList<>();
        for (int k = 0; k < clusterNum; k++) {
            Instances dataFliter = Cluster_Fliter.filter(data,k);
            int[] listList = new int[numLabels];
            for (int j = 0; j < dataFliter.numInstances(); j++) {
                for (int i = 0; i < numLabels; i++) {
                    listList[i] -= (int) dataFliter.get(j).value(i);
                }
            }
            List<Integer> ListOfInt = new ArrayList<>();
            List<Integer> ListOfNonInt = new ArrayList<>();
            for (int i = 0; i < listList.length; i++) {
//                List<Integer> ListOfInt = new ArrayList<>();
                if(listList[i]<0){
                    ListOfInt.add(i);
                }else{
                    ListOfNonInt.add(i);
                }
            }
            Integer[] intArrayLabels = new Integer[ListOfInt.size()];
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            Integer[] intArrayNonLabels = new Integer[ListOfNonInt.size()];
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            intArrayNonLabels = ListOfNonInt.toArray(intArrayNonLabels);
            listOfLabels.add(intArrayLabels);
            listOfNonLabels.add(intArrayNonLabels);
            Instances dataFliter1 = Cluster_Fliter.filter(data,1);
            for (Integer integer : listOfNonLabels.get(1)) {
                dataFliter1.deleteAttributeAt(integer);
            }
            System.out.println(dataFliter1);
//            System.out.println(Arrays.toString(intArrayLabels));
        }
        return listOfLabels;
    }
    public static void main(String[] args) throws Exception {
//        long time1= System.nanoTime();
//        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
//        for (int i = 0; i < 8; i++) {
//            CC_test.ccRun(i,66,"src/main/CAL500_clustered_adjusted.arff",null);
//        }
//        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
//        System.out.println(time2);
//       int[] label = new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 43, 53, 56, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 79, 81, 82, 84, 86, 87, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 120, 121, 124, 125, 126, 130, 136, 138, 141, 143, 147, 148, 149, 150, 157, 160, 161, 164, 166, 170, 172};
//        CC_test.ccRun(2,66,"src/main/CAL500_clustered_adjusted.arff",label);


        

//
//        INDArray nd = Nd4j.create(new float[]{1, 2, 3, 4}, new int[]{2, 2});
//        INDArray nd1 = Nd4j.create(new float[]{1, 2}, new int[]{2});
//        System.out.println(nd);
//        System.out.println(nd.addRowVector(nd1));


        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
        Instances data = source.getDataSet();

        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if(matcher.find()){
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }

        List<Integer[]> listOfLabels= new ArrayList<>();
        List<Integer[]> listOfNonLabels = new ArrayList<>();
        for (int k = 0; k < 8; k++) {
            Instances dataFliter = Cluster_Fliter.filter(data,k);
            int[] listList = new int[numLabels];
            for (int j = 0; j < dataFliter.numInstances(); j++) {
                for (int i = 0; i < numLabels; i++) {
                    listList[i] -= (int) dataFliter.get(j).value(i);
                }
            }
            List<Integer> ListOfInt = new ArrayList<>();
            List<Integer> ListOfNonInt = new ArrayList<>();
            for (int i = 0; i < listList.length; i++) {
//                List<Integer> ListOfInt = new ArrayList<>();
                if(listList[i]<0){
                    ListOfInt.add(i);
                }else{
                    ListOfNonInt.add(i);
                }
            }
            Integer[] intArrayLabels = new Integer[ListOfInt.size()];
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            Integer[] intArrayNonLabels = new Integer[ListOfNonInt.size()];
            intArrayLabels = ListOfInt.toArray(intArrayLabels);
            intArrayNonLabels = ListOfNonInt.toArray(intArrayNonLabels);
            listOfLabels.add(intArrayLabels);
            listOfNonLabels.add(intArrayNonLabels);


//            System.out.println(dataFliter1);
//            System.out.println(Arrays.toString(intArrayLabels));
        }
        Instances dataFliter1 = Cluster_Fliter.filter(data,1);
        System.out.println(Arrays.toString(listOfNonLabels.get(1)));
        for (Integer integer : listOfNonLabels.get(1)) {
            dataFliter1.deleteAttributeAt(integer);
        }
//         = System.out.println(dataFliter1);
    }
}
