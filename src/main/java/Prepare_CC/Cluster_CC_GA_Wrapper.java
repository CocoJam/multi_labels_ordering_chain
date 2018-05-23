package Prepare_CC;

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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Cluster_CC_GA_Wrapper {
    public Instances clustered;
    public int clusterNumber;
    public ClusterEvaluation evaluation;
    public List<Cluster_CC_Builder> listOfClusterBuilder = new ArrayList<>();
    public Instances test;
    public Instances train;
    public String string="";
    public Cluster_CC_GA_Wrapper(String file,int seed) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances data = source.getDataSet();
        EM EM_test = new EM();
        //Seeding for eval
        EM_test.setSeed(seed);

        EM_test.buildClusterer(data);
        //eval object for any clusters
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(EM_test);
        eval.evaluateClusterer(data);
        evaluation = eval;
        clusterNumber = eval.getNumClusters();
        BufferedWriter bufferedWriter;
        bufferedWriter = new BufferedWriter(new FileWriter(new File("T_"+seed+"/Clustering_Log")));
//        System.out.println(eval.clusterResultsToString());
        bufferedWriter.write(eval.clusterResultsToString());
        bufferedWriter.close();
        //Print the overall results
        Instances newData = new Instances(data);
        Attribute attribute = new Attribute("Cluster");
        newData.insertAttributeAt(attribute, newData.numAttributes());
        for (int i = 0; i < newData.numInstances(); i++) {
            System.out.println(newData.get(i).attribute(data.get(i).numAttributes()));
            newData.get(i).setValue(newData.get(i).attribute(newData.get(i).numAttributes() - 1), EM_test.clusterInstance(data.get(i)));
        }
        this.clustered = newData;
        System.out.println("Clustered");
        ArffSaver saver = new ArffSaver();
//        System.out.println(newData);
        saver.setInstances(newData);
        saver.setFile(new File("src/main/trail_1.arff"));
        saver.writeBatch();
        for (int i = 0; i < eval.getNumClusters(); i++) {
            Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(i, newData, 0);
            if (cluster_cc_builder.cluster.numInstances() >= newData.numInstances() * 0.1) {
                listOfClusterBuilder.add(cluster_cc_builder);
            } else {
                string+=("Missed cluster: " + i+"\n");
            }

        }
    }

    public List<Feature_Vector> runAndReturn(List<Cluster_CC_Builder> listOfClusterBuilder) throws Exception {
        GA_CC ga_cc = null;
        List<Feature_Vector> feature_vectors = new ArrayList<>();
        List<GA_CC> ga_ccs = new ArrayList<>();
        for (Cluster_CC_Builder cluster_cc_builder : listOfClusterBuilder) {
            System.out.println("Building cluster cc builder");
            try {
                ga_cc = GA_CC.of(cluster_cc_builder, 20, 10);
            } catch (IOException e) {
                e.printStackTrace();
            }
            ga_cc.thread.start();
            ga_ccs.add(ga_cc);
        }
        for (GA_CC ga_cc1 : ga_ccs) {
            ga_cc1.thread.join();
            int[] trainedChain = ga_cc1.trainedChain;
            feature_vectors.add(new Feature_Vector(ga_cc1.cluster_cc_builder.featureVector, ga_cc1.cluster_cc_builder.clusterNum, ga_cc1.cluster_cc_builder.labelChain, trainedChain, ga_cc1.cluster_cc_builder.overallLabelNum, ga_cc1.cluster_cc_builder.cluster));
        }
        return feature_vectors;
    }

    public List<Result> ResultsAndEvalution(List<Cluster_CC_Builder> listOfClusterBuilder) throws Exception {
        List<Result> results = new ArrayList<>();
        GA_CC ga_cc = null;
        List<Feature_Vector> feature_vectors = new ArrayList<>();
        List<GA_CC> ga_ccs = new ArrayList<>();
        for (Cluster_CC_Builder cluster_cc_builder : listOfClusterBuilder) {
            System.out.println("Building cluster cc builder");
            try {
                ga_cc = GA_CC.of(cluster_cc_builder, 20, 10);
            } catch (IOException e) {
                e.printStackTrace();
            }
            ga_cc.thread.start();
            ga_ccs.add(ga_cc);
        }
        for (GA_CC ga_cc1 : ga_ccs) {
            ga_cc1.thread.join();
            String top = "PCut1";
            String vop = "3";
            Cluster_CC_Builder cluster_cc_builder = ga_cc1.cluster_cc_builder;
            Base_CC cc = new Base_CC();
            cc.prepareChain(ga_cc1.trainedChain);
            MLUtils.prepareData(cluster_cc_builder.parsedCluster);
            cc.buildClassifier(cluster_cc_builder.parsedCluster);
            int numOfCV = cluster_cc_builder.parsedCluster.numInstances() > 10 ? 10 : cluster_cc_builder.parsedCluster.numInstances();
            try {
                Result result = Evaluation.cvModel(cc, ga_cc1.cluster_cc_builder.parsedCluster, numOfCV, top, vop);
                results.add(result);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
        return results;
    }



    public List<Result> EvaluationFeatureVector(List<Feature_Vector> feature_vectors) {
        List<Result> results = new ArrayList<>();
        for (Feature_Vector feature_vector : feature_vectors) {
            String top = "PCut1";
            String vop = "3";
            int numOfCV = feature_vector.instances.numInstances() > 10 ? 10 : feature_vector.instances.numInstances();
            try {
                Result result = Evaluation.cvModel(feature_vector.base_cc, feature_vector.instances, numOfCV, top, vop);
                results.add(result);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("Result returning");
        return results;
    }

    public static void main(String[] args) throws Exception {
        String overallCSV = "Hamming_Loss,Exact_match,Accuracy,Average";
        for (int j = 0; j < 10; j++) {
            File file = new File("T_"+j);
            if (!file.exists()){
                file.mkdir();
            }
            Cluster_CC_GA_Wrapper cluster_cc_ga_wrapper = new Cluster_CC_GA_Wrapper("src/main/CAL500_train.arff",j);
//            List<Result> results = cluster_cc_ga_wrapper.EvaluationFeatureVector(cluster_cc_ga_wrapper.runAndReturn(cluster_cc_ga_wrapper.listOfClusterBuilder));
            List<Result> results = cluster_cc_ga_wrapper.ResultsAndEvalution(cluster_cc_ga_wrapper.listOfClusterBuilder);
            double overallHamming_loss=0;
            double overallExact_match=0;
            double overallAccuracy = 0;
            double overallAverage = 0;
            System.out.println(results.size());
            for (int i = 0; i < results.size(); i++) {
                Result result = results.get(i);
                System.out.println("");
                double hamming_loss = Double.parseDouble(result.getMeasurement("Hamming loss").toString());
                double exact_match = Double.parseDouble(result.getMeasurement("Exact match").toString());
                double accuracy = Double.parseDouble(result.getMeasurement("Accuracy").toString());
                overallExact_match+=exact_match/results.size();
                overallHamming_loss+=hamming_loss/results.size();
                overallAccuracy+=accuracy/results.size();

                double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;
                overallAverage+= averaging/results.size();
                BufferedWriter bufferedWriter;
                bufferedWriter = new BufferedWriter(new FileWriter(new File("T_"+j+"/Logging_"+i+"_results")));
                bufferedWriter.write(result.toString());
                bufferedWriter.write("Averaging: "+averaging);
                bufferedWriter.close();
            }
            BufferedWriter bufferedWriter;
            bufferedWriter = new BufferedWriter(new FileWriter(new File("T_"+j+"/Logging_OverallResults")));
            bufferedWriter.write(cluster_cc_ga_wrapper.string);
            bufferedWriter.write("Hamming_loss: "+overallHamming_loss+"\n");
            bufferedWriter.write("Exact_match: "+overallExact_match+"\n");
            bufferedWriter.write("Accuracy: "+overallAccuracy+"\n");
            bufferedWriter.write("Averaging: "+overallAverage);
            bufferedWriter.close();
            overallCSV += overallAverage+","+overallExact_match+","+overallHamming_loss+","+overallAccuracy+"\n";
        }
        BufferedWriter bufferedWriter;
        bufferedWriter = new BufferedWriter(new FileWriter(new File("Overall.csv")));
        bufferedWriter.write(overallCSV);
        bufferedWriter.close();
    }
}
