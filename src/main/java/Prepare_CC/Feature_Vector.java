package Prepare_CC;

import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.core.MLUtils;
import weka.classifiers.lazy.IBk;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

import java.io.File;
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
//    public List<>

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
//        for (int i = 0; i < 10; i++) {
//            ConverterUtils.DataSource source = new ConverterUtils.DataSource("src/main/CAL500_train.arff");
//            Instances data = source.getDataSet();
//            EM EM_test = new EM();
//            //Seeding for eval
//            EM_test.setSeed(i);
//
//            EM_test.buildClusterer(data);
//            //eval object for any clusters
//            ClusterEvaluation eval = new ClusterEvaluation();
//            eval.setClusterer(EM_test);
//            eval.evaluateClusterer(data);
//            Instances newData = new Instances(data);
//            Attribute attribute = new Attribute("Cluster");
//            newData.insertAttributeAt(attribute, newData.numAttributes());
//
//            for (int j = 0; j < newData.numInstances(); j++) {
//                System.out.println(newData.get(j).attribute(data.get(j).numAttributes()));
//                newData.get(j).setValue(newData.get(j).attribute(newData.get(j).numAttributes() - 1), EM_test.clusterInstance(data.get(j)));
//            }
//
//            Pattern pattern = Pattern.compile("((.+-C )(\\d+))");
//            Matcher matcher = pattern.matcher(data.relationName());
//            int numLabels = 0;
//            String group2 = "";
//            if (matcher.find()) {
//                data.setRelationName(matcher.group(0));
//                group2 = matcher.group(2);
//                numLabels = Integer.parseInt(matcher.group(3));
//            }
//            File file = new File("Train_Cluster_"+i);
//            if (!file.exists()){
//                file.mkdir();
//            }
//            ArffSaver saver = new ArffSaver();
////        System.out.println(newData);
//            saver.setInstances(newData);
//            saver.setFile(new File(file.getCanonicalPath()+"/"+i));
//            saver.writeBatch();
//        }



    }
}
