package Prepare_CC;

import meka.classifiers.multilabel.CC;
import meka.core.MLUtils;
import weka.classifiers.lazy.IBk;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class Feature_Vector {


    public double[] fectureVector;
    public int clusterNum;
    public int[] orginalChain;
    public int[] trainedLabelChain;
    public int[] preparedTrainedLabelChain;
    public Instances instances;
    public Base_CC base_cc;

    public Feature_Vector(double[] fectureVector, int clusterNum, int[] orginalChain, int[] trainedLabelChain, int orginalLabelNum, Instances instances) throws Exception {

        this.fectureVector = fectureVector;
        this.clusterNum = clusterNum;
        this.orginalChain = orginalChain;
        this.trainedLabelChain = trainedLabelChain;
        this.preparedTrainedLabelChain = preparedTrainedLabelChain(trainedLabelChain, orginalChain,orginalLabelNum);
        base_cc = new Base_CC();

        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(instances.relationName());
        int numLabels = 0;
        if (matcher.find()) {
            instances.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        this.instances = instances;
        base_cc.prepareChain(this.preparedTrainedLabelChain);
        MLUtils.prepareData(instances);
        base_cc.buildClassifier(instances);
    }
    public int[] preparedTrainedLabelChain(int[] trained, int[] orginalChain, int orginalLabelNum){
        int[] preparedLabelChain = new int[trained.length];
        int[] orginalLabelSet = new int[orginalLabelNum];
        for (int i = 0; i < orginalLabelNum; i++) {
            orginalLabelSet[i] = i;
        }
        System.out.println(Arrays.toString(orginalChain));
        System.out.println(orginalChain.length);
        System.out.println(Arrays.toString(trained));
        System.out.println(trained.length);
        for (int i = 0; i < trained.length; i++) {
            int tr = trained[i];
            int or = orginalChain[tr];
            preparedLabelChain[i] = or;
        }
        List<Integer> UnionSet = new ArrayList<>();
        for (int i : preparedLabelChain) {
            UnionSet.add(i);
        }
        for (int i : orginalLabelSet) {
            if (!UnionSet.contains(i)){
                UnionSet.add(i);
            }
        }
        return  Arrays.stream(UnionSet.toArray(new Integer[UnionSet.size()])).mapToInt(Integer::intValue).toArray();
    }

    public static void main(String[] args) throws Exception {


    }
}
