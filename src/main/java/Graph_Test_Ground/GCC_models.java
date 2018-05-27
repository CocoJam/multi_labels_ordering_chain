package Graph_Test_Ground;

import Prepare_CC.Base_CC;
import Prepare_CC.Cluster_CC_Builder;
import Prepare_CC.Cluster_CC_GA_Wrapper;
import WEKA_Test_Ground.CAL500_TEST_Train_SPLIT;
import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.Evaluation;
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


        for (int i = 0; i < 10; i++) {
            List<int[]> results = new ArrayList<>();
            long time1 = System.nanoTime();
            Instances train = (new ConverterUtils.DataSource("Split_" + i + "/CAL500_train.arff")).getDataSet();
            Instances test = (new ConverterUtils.DataSource("Split_" + i + "/CAL500_test.arff")).getDataSet();
            int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount - 1;
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
            for (int j = 0; j < results.size(); j++) {
                ClusterTracking +="Trial,"+i+ ",Cluster,"+j+",best result chain,"+ Arrays.toString(results.get(j))+"\n";
            }

            double overallExact_match= 0;
            double overallHamming_loss=0;
            double overallAccuracy=0;
            double overallAverage=0;
            Instances testInstances = Cluster_Fliter.knn_inference(train, test, 5);
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
                cc.prepareChain(results.get(j));
                cc.buildClassifier(cluster_cc_builder.parsedCluster);

                String top = "PCut1";
                String vop = "3";
                Result evaluateModel;
                try{
                evaluateModel = Evaluation.evaluateModel(cc, cluster_cc_builder.parsedCluster, clusterX, top, vop);}catch (ArrayIndexOutOfBoundsException e){
                    continue;
                }

                double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
                overallExact_match+=exact_match/ numberOfCluster;
                overallHamming_loss+=hamming_loss/ numberOfCluster;
                overallAccuracy+=accuracy/ numberOfCluster;

                double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;
                overallAverage+= averaging/results.size();
            }
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("graph_test_result/T_"+i+".csv")));
            bufferedWriter.write(ClusterTracking);
            bufferedWriter.write("Hamming_loss, "+overallHamming_loss+"\n");
            bufferedWriter.write("Exact_match, "+overallExact_match+"\n");
            bufferedWriter.write("Accuracy, "+overallAccuracy+"\n");
            bufferedWriter.write("Averaging, "+overallAverage);
            bufferedWriter.close();
            System.out.println("One trial: ");
            System.out.println(System.nanoTime()-time1);

        }


    }
}
