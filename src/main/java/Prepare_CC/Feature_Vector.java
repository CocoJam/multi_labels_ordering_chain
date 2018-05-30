package Prepare_CC;

import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import weka.classifiers.lazy.IBk;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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
        this.preparedTrainedLabelChain = preparedTrainedLabelChain(trainedLabelChain, orginalChain, orginalLabelNum);
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

    public int[] preparedTrainedLabelChain(int[] trained, int[] orginalChain, int orginalLabelNum) {
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
            if (!UnionSet.contains(i)) {
                UnionSet.add(i);
            }
        }
        return Arrays.stream(UnionSet.toArray(new Integer[UnionSet.size()])).mapToInt(Integer::intValue).toArray();
    }


    public static void main(String[] args) throws Exception {

        for (int w = 0; w < 10; w++) {
            String Tracking = "Sample,Hamming_loss,Exact_match,Accuracy,\n";
            int sampleNumber = 1;
            List<Double> ham = new ArrayList<>();
            List<Double> exact = new ArrayList<>();
            List<Double> acc = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                long time1 = System.nanoTime();
                Instances train = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
                Instances test = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
                int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount;
//        System.out.println(train.attributeStats(train.numAttributes()-1).distinctCount);
                List<Cluster_CC_Builder> cluster_cc_builders = new ArrayList<>();
                String ClusterTracking = "";
                for (int j = 0; j < numberOfCluster; j++) {
                    Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                    cluster_cc_builders.add(cluster_cc_builder);
                }
                List<GA_CC> results = Cluster_CC_GA_Wrapper.ResultsChainsDropLabel(cluster_cc_builders);
                Instances testInstances = Cluster_Fliter.knn_inference(train, test, 3);
                List<Result> resultsList = new ArrayList<>();
                for (int j = 0; j < numberOfCluster; j++) {
                    Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                    Instances clusterX = Cluster_Fliter.filter(testInstances, j);
                    Remove remove = new Remove();
                    remove.setAttributeIndicesArray(cluster_cc_builder.labelsDropped);
                    remove.setInputFormat(clusterX);
                    clusterX = Filter.useFilter(clusterX, remove);
                    Pattern pattern = Pattern.compile("(.+-C (\\d+))");
                    Matcher matcher = pattern.matcher(clusterX.relationName());
                    if (matcher.find()) {
                        clusterX.setRelationName(cluster_cc_builder.parsedCluster.relationName());
                    }
                    Base_CC cc = new Base_CC();
                    MLUtils.prepareData(cluster_cc_builder.parsedCluster);
                    MLUtils.prepareData(clusterX);
                    cc.prepareChain(results.get(j).trainedChain);
                    cc.buildClassifier(cluster_cc_builder.parsedCluster);
                    String top = "PCut1";
                    String vop = "3";
                    Result evaluateModel;
                    try {
                        evaluateModel = Evaluation.evaluateModel(cc, cluster_cc_builder.parsedCluster, clusterX, top, vop);
                        resultsList.add(evaluateModel);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        System.out.println(e);
                        continue;
                    }
                    Tracking += sampleNumber + ",";
                    sampleNumber++;
                    double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                    ham.add(hamming_loss);
                    Tracking += hamming_loss + ",";
                    double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                    exact.add(exact_match);
                    Tracking += exact_match + ",";
                    double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
                    acc.add(accuracy);
                    Tracking += accuracy + ",\n";
//                System.out.println(evaluateModel);
                }



//            System.out.println(Arrays.toString(cluster_cc_builder.sqeuenceChain));
            }
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("CVSeed_" + w + "/ LABEL_DROP_CVSPLIT_GA_Results_CV_5_20iteration_10_Pop_Mut_025_Tournament_popDiv4_Elit_4_K_3.csv")));
            double ham_summ = ham.stream().reduce(0.0, Double::sum);
            double exact_summ = exact.stream().reduce(0.0, Double::sum);
            double acc_summ = acc.stream().reduce(0.0, Double::sum);



            double ham_average = ham_summ / sampleNumber;
            double exact_average = exact_summ / sampleNumber;
            double acc_average = acc_summ / sampleNumber;

            double ham_var = ham.stream().reduce(0.0, (x, y) -> x + Math.pow((y - ham_average), 2));
            double exact_var = exact.stream().reduce(0.0, (x, y) -> x + Math.pow((y - exact_average), 2));
            double acc_var = acc.stream().reduce(0.0, (x, y) -> x + Math.pow((y - acc_average), 2));
            Tracking += "Average," + ham_average + "," + exact_average + "," + acc_average + ",\n";
            Tracking += "varience," + ham_var / sampleNumber + "," + exact_var / sampleNumber + "," + acc_var / sampleNumber + ",\n";
            Tracking += "standard deviation," + Math.sqrt(ham_var / sampleNumber) + "," + Math.sqrt(exact_var / sampleNumber) + "," + Math.sqrt(acc_var / sampleNumber) + ",\n";
            try {

                bufferedWriter.write(Tracking);
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
