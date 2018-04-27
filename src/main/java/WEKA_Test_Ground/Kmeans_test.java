package WEKA_Test_Ground;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.CoverTree;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;


public class Kmeans_test {
    public static void main(String[] args) {
        //CSVLoader parse given csv file to arff which is what WEKA uses.
        CSVLoader loader = new CSVLoader();
        try {
//            File file = new File ("src/main/Mediamill_joint.csv");
            File file = new File ("src/main/Mediamill_joint.csv");
//            File file = new File ("src/main/Mediamill_labelset");
            if (!file.exists())
                System.exit(0);
            loader.setFile(file);
            //This is the method to read the file and parse everything in to instances
            Instances data = loader.getDataSet();
            //This is how u access the instances by get(int) and .attribute(int)
//            System.out.println(data.get(0).attribute(0));
//            System.out.println(data.get(0).attribute(1));

            //Set up the kmeans cluster
            SimpleKMeans kmeans = new SimpleKMeans();
            //Seeding for eval
            kmeans.setSeed(10);

            //Keep instance order since it is important for kmeans cluster
            kmeans.setPreserveInstancesOrder(true);
            //Cluster size
            kmeans.setNumClusters(10);
            kmeans.buildClusterer(data);

            //eval object for any clusters
            ClusterEvaluation eval = new ClusterEvaluation();
            eval.setClusterer(kmeans);
            eval.evaluateClusterer(data);

            //Print the overall results
            System.out.println(eval.clusterResultsToString());

            //To get individual instances which cluster did it went into if instance 1 is in cluster 5 then
            //assignments[0] = 5

            int[] assignments = kmeans.getAssignments();
            int i=0;

            Map<Integer,Integer> clusterInstances = new TreeMap<Integer, Integer>();
            for(int clusterNum : assignments) {
                System.out.println("Instance -> Cluster "+ i + "->"+ clusterNum);
                if (clusterInstances.get(clusterNum) ==null){
                    System.out.println(clusterNum + " is new so 1");
                    clusterInstances.put(clusterNum,1);
                }
                else{
                    int num =clusterInstances.get(clusterNum);
                    System.out.println(clusterNum + " it has "+ num);
                    clusterInstances.put(clusterNum,num+1);
                }
                i++;
            }
        } catch (IOException e) {
            System.out.println(e);
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
