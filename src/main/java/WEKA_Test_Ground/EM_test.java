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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.core.converters.ConverterUtils.DataSource;



public class EM_test {


    //try again
    public static void main(String[] args) throws Exception {
        try {
            long time1 = System.nanoTime();
            DataSource source = new DataSource("src/main/mediamill.arff");
            Instances data = source.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
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
            System.out.println( EM_test.clusterInstance(data.get(1)));
            Instances newData = new Instances(data);
            Attribute attribute = new Attribute("Cluster");
            newData.insertAttributeAt(attribute,newData.numAttributes());
            for (int i = 0; i < newData.numInstances(); i++) {
                System.out.println(newData.get(i).attribute(data.get(i).numAttributes()));
//                System.out.println(EM_test.clusterInstance(newData.get(i)));
                newData.get(i).setValue(newData.get(i).attribute(newData.get(i).numAttributes()-1), EM_test.clusterInstance(data.get(i)));
            }
            ArffSaver saver = new ArffSaver();
            System.out.println(newData);
            saver.setInstances(newData);
            saver.setFile(new File("src/main/medialmill_clustered_adjusted.arff"));
            saver.writeBatch();

        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
