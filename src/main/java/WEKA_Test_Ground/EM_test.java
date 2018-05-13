package WEKA_Test_Ground;

/**
 * Created by Administer on 1/05/2018.
 */

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.neighboursearch.CoverTree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import weka.core.converters.ConverterUtils.DataSource;



public class EM_test {


    //try again
    public static void main(String[] args) {

        try {
            DataSource source = new DataSource("src/main/CAL500.arff");
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

            //Print the overall results
            System.out.println("# of clusters: " + eval.getNumClusters());
            System.out.println(eval.clusterResultsToString());




        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
