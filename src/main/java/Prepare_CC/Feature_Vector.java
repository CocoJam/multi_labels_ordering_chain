package Prepare_CC;

import meka.classifiers.multilabel.CC;
import meka.core.MLUtils;
import weka.classifiers.lazy.IBk;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.*;


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
        this.instances = instances;
        MLUtils.prepareData(instances);
        base_cc.buildClassifier(instances);
    }
    public int[] preparedTrainedLabelChain(int[] trained, int[] orginalChain, int orginalLabelNum){
        int[] preparedLabelChain = new int[orginalLabelNum];
        int[] orginalLabelSet = new int[orginalLabelNum];
        for (int i = 0; i < orginalLabelNum; i++) {
            orginalLabelSet[i] = i;
        }
        for (int i = 0; i < preparedLabelChain.length; i++) {
            preparedLabelChain[i] = orginalChain[trained[i]];
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
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_clustered_adjusted.arff");
        Instances data = source.getDataSet();

        double split = 66;
        int trainSize = (int) (data.numInstances() * split / 100.0);
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, data.numInstances() - trainSize);

    }
}
