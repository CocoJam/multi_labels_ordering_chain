package Graph_Test_Ground;

import Prepare_CC.Cluster_CC_Builder;
import WEKA_Test_Ground.Cluster_Fliter;
import org.ejml.simple.SimpleMatrix;
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.SingleGraph;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;

/**
 * Created by Administer on 19/05/2018.
 */



public class cluster_trial {
    public static void main(String[] args) throws Exception {


        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff",3,0);

        int[] labels = cluster_cc_builder.labelChain;
        System.out.println("Label chain: "+ labels.length);
        int[] seqIndex = cluster_cc_builder.sqeuenceChain;
        Instances cluster = cluster_cc_builder.cluster;
        int intNum = cluster.numInstances();
        int labNum = labels.length;
        ArrayList<double[]> aug_inst = new ArrayList<>();



        //preparing instances
        for (int i = 0; i < intNum; i++){
            double[] instValues = cluster.instance(i).toDoubleArray();
            double[] aug_array = new double[labNum];
            for (int j = 0; j <labNum; j++){
                aug_array[j]= instValues[labels[j]];
            }
            aug_inst.add(aug_array);
        }

        //loading cluster instances into matrix
        SimpleMatrix cluser_max = new SimpleMatrix(intNum, labNum);
        for(int i =0; i <intNum; i++) {
            cluser_max.setRow(i,0, aug_inst.get(i));
        }


        // Construct co-occurence matrix
        SimpleMatrix co_max = new SimpleMatrix(labNum, labNum);
        for (int r =0; r<labNum;r++){
            for(int c = 0; c < labNum; c++){
                if (c == r ){
                    co_max.set(r,c,0);
                }else{
                 SimpleMatrix rVec = cluser_max.extractVector(false, r);
                 SimpleMatrix cVec = cluser_max.extractVector(false, c);
                 double count = rVec.elementMult(cVec).elementSum();
                 co_max.set(r,c,count);
                }
            }
        }
        System.out.println(co_max);


//        Graph graph = new SingleGraph(cluster_cc_builder.clusterNum+"");
//        graph.setStrict(false);
//        graph.setAutoCreate( true );
//
//        for (int index: labels) {
//
//        }

    }
}
