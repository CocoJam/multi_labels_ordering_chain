package Graph_Test_Ground;

import Prepare_CC.Base_CC;
import Prepare_CC.Cluster_CC_Builder;
import Prepare_CC.Cluster_CC_GA_Wrapper;
import WEKA_Test_Ground.CAL500_TEST_Train_SPLIT;
import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.Evaluation;
import meka.core.A;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by Administer on 24/05/2018.
 */
public class GCC_models {

    public static void main(String[] args) throws Exception {

        List<Double> hamming = new ArrayList<>();
        List<Double> Exact = new ArrayList<>();
        List<Double> Acc =new ArrayList<>();
        int sample = 1;
        String tracking = "Sample,Hamming_loss, Exact_match, Acc,\n";
        for (int i = 0; i < 10; i++) {
            List<int[]> results = new ArrayList<>();
            long time1 = System.nanoTime();
            Instances train = (new ConverterUtils.DataSource("CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
            Instances test = (new ConverterUtils.DataSource("CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
            int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount;
//        System.out.println(train.attributeStats(train.numAttributes()-1).distinctCount);
            //List<Cluster_CC_Builder> cluster_cc_builders = new ArrayList<>();
            String ClusterTracking = "";
            for (int j = 0; j < numberOfCluster; j++) {
                Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                Graph_processing model = new Graph_processing(cluster_cc_builder);
                int[] optimized = model.optimize_order();
                results.add(optimized);
                //cluster_cc_builders.add(cluster_cc_builder);
            }



            //List<int[]> results = Cluster_CC_GA_Wrapper.ResultsChains(cluster_cc_builders);
            System.out.println("Start");
            System.out.println(System.nanoTime()-time1);


            double overallExact_match= 0;
            double overallHamming_loss=0;
            double overallAccuracy=0;
            double overallAverage=0;
            Instances testInstances = Cluster_Fliter.knn_inference(train, test, 3);
            for (int j = 0; j < numberOfCluster; j++) {
                Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                Instances clusterX = Cluster_Fliter.filter(testInstances, j);
//                Remove remove = new Remove();
//                remove.setAttributeIndicesArray(cluster_cc_builder.labelsDropped);
//                remove.setInputFormat(clusterX);
//                clusterX = Filter.useFilter(clusterX, remove);
//                Pattern pattern = Pattern.compile("(.+-C (\\d+))");
//                Matcher matcher = pattern.matcher(clusterX.relationName());
//                if (matcher.find()) {
//                    clusterX.setRelationName(cluster_cc_builder.parsedCluster.relationName());
//                }
                Base_CC cc = new Base_CC();
                MLUtils.prepareData(cluster_cc_builder.cluster);
                MLUtils.prepareData(clusterX);
                System.out.println(cluster_cc_builder.cluster.relationName());
                System.out.println(clusterX.numAttributes());
                cc.prepareChain(results.get(j));
                System.out.println(results.get(j).length);
                cc.buildClassifier(cluster_cc_builder.cluster);

                String top = "PCut1";
                String vop = "3";
                Result evaluateModel;
                try{
                evaluateModel = Evaluation.evaluateModel(cc, cluster_cc_builder.cluster, clusterX, top, vop);}catch (ArrayIndexOutOfBoundsException e){
                    continue;
                }
                tracking += sample+",";
                sample ++;
                double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());

                tracking += hamming_loss +",";
                hamming.add(hamming_loss);
                tracking += exact_match +",";
                Exact.add(exact_match);
                tracking += accuracy+",\n";
                Acc.add(accuracy);
            }


        }
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("nondrop_graph_test_result/results.csv")));


        double ham_summ = hamming.stream().reduce(0.0, Double::sum);
        double exact_summ = Exact.stream().reduce(0.0, Double::sum);
        double acc_summ = Acc.stream().reduce(0.0, Double::sum);

        double ham_average = ham_summ / sample;
        double exact_average = exact_summ / sample;
        double acc_average = acc_summ / sample;

        double ham_var = hamming.stream().reduce(0.0, (x, y) -> x + Math.pow((y - ham_average), 2));
        double exact_var = Exact.stream().reduce(0.0, (x, y) -> x + Math.pow((y - exact_average), 2));
        double acc_var = Acc.stream().reduce(0.0, (x, y) -> x + Math.pow((y - acc_average), 2));
        tracking += "Average," + ham_average + "," + exact_average + "," + acc_average + ",\n";
        tracking += "varience," + ham_var / sample + "," + exact_var / sample + "," + acc_var / sample + ",\n";
        tracking += "standard deviation," + Math.sqrt(ham_var / sample) + "," + Math.sqrt(exact_var / sample) + "," + Math.sqrt(acc_var / sample) + ",\n";
        bufferedWriter.write(tracking);
        bufferedWriter.close();
        System.out.println("One trial: ");
//        System.out.println(System.nanoTime()-time1);
    }
}
