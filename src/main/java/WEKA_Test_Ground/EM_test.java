package WEKA_Test_Ground;

/**
 * Created by Administer on 1/05/2018.
 */

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.CoverTree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import weka.core.converters.ConverterUtils.DataSource;



public class EM_test {


    //try again
    public static void main(String[] args) throws Exception {
        try {
            long time1 = System.nanoTime();
            DataSource source = new DataSource("src/main/mediamil_adjusted.arff");
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well

//            if (data.classIndex() == -1){
//                data.setClassIndex(data.numAttributes() - 1);
//            };
//            System.out.println(data.numAttributes());

            //Set up the EM_test cluster
            EM EM_test = new EM();
            //Seeding for eval
            EM_test.setSeed(10);


            EM_test.buildClusterer(data);

            //eval object for any clusters
            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(EM_test);
            eval.evaluateClusterer(data);
            EM_test.getCapabilities();
            //Print the overall results
            System.out.println("# of clusters: " + eval.getNumClusters());
            System.out.println(eval.clusterResultsToString());
            long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
            System.out.println( time2);
            SimpleKMeans kmeans = new SimpleKMeans();
            //Seeding for eval
            kmeans.setSeed(10);
            kmeans.setPreserveInstancesOrder(true);
            kmeans.setNumClusters(eval.getNumClusters());
            kmeans.buildClusterer(data);
            ClusterEvaluation eval2 = new ClusterEvaluation();
            eval2.setClusterer(kmeans);
            eval2.evaluateClusterer(data);

            //Print the overall results
            System.out.println(eval2.clusterResultsToString());
            int[] assignments = kmeans.getAssignments();
            int i=0;

//            Map<Integer,Integer> clusterInstances = new TreeMap<Integer, Integer>();
            Instances newData = new Instances(data);
            Attribute attribute = new Attribute("Cluster");
            newData.insertAttributeAt(attribute,0);
            for(int clusterNum : assignments) {
                System.out.println("Instance -> Cluster "+ i + "->"+ clusterNum);
                Instance j= newData.get(i);
                j.setValue(j.attribute(j.numAttributes()-1),clusterNum);
                System.out.println(j);
                i++;
            }
            ArffSaver saver = new ArffSaver();
            saver.setInstances(newData);
            saver.setFile(new File("src/main/mediamil_clustered_adjusted.arff"));
//            saver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
            saver.writeBatch();

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
