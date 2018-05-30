package Graph_Test_Ground;

import Prepare_CC.Base_CC;
import Prepare_CC.Cluster_CC_Builder;
import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import org.ejml.simple.SimpleMatrix;
import org.graphstream.algorithm.*;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.Path;
import org.graphstream.graph.implementations.SingleGraph;
import org.kramerlab.bmad.algorithms.Iter;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.lang.reflect.Array;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.lang.Float.NaN;

/**
 * Created by Administer on 19/05/2018.
 */



public class cluster_trial {
    public static void main(String[] args) throws Exception {


        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff", 4, 0);
        Instances cluster = cluster_cc_builder.cluster;
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(cluster.relationName());
        int numLabels = 0;
        if (matcher.find()) {
            cluster.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
            System.out.println(numLabels);
        }
        int[] labels =cluster_cc_builder.labelChain;
//        for (int i = 0; i < numLabels; i++) {
//            labels[i] =i;
//        }
//        int[] labels = cluster_cc_builder.labelChain;
        System.out.println("Label chain: " + labels.length);
        int[] seqIndex = cluster_cc_builder.sqeuenceChain;

        int intNum = cluster.numInstances();
        int labNum = labels.length;
        int[] finalisedOrder = new int[labNum];
        //final double thres = 0.2;
        ArrayList<double[]> aug_inst = new ArrayList<>();


        //preparing instances
        for (int i = 0; i < intNum; i++) {
            double[] instValues = cluster.instance(i).toDoubleArray();
            double[] aug_array = new double[labNum];
            for (int j = 0; j < labNum; j++) {
                aug_array[j] = instValues[labels[j]];
            }
            aug_inst.add(aug_array);
        }

        //loading cluster instances into matrix
        SimpleMatrix cluser_max = new SimpleMatrix(intNum, labNum);
        for (int i = 0; i < intNum; i++) {
            cluser_max.setRow(i, 0, aug_inst.get(i));
        }


        // Construct co-occurence matrix
        SimpleMatrix co_max = new SimpleMatrix(labNum, labNum);
        for (int r = 0; r < labNum; r++) {
            for (int c = 0; c < labNum; c++) {
                if (c == r) {
                    co_max.set(r, c, 0);
                } else {
                    SimpleMatrix rVec = cluser_max.extractVector(false, r);
                    SimpleMatrix cVec = cluser_max.extractVector(false, c);
                    double count = rVec.elementMult(cVec).elementSum();
                    co_max.set(r, c, count);
                }
            }
        }
        System.out.println(co_max);


        //Create graph
        Graph graph = new SingleGraph(cluster_cc_builder.clusterNum + "");
        graph.setStrict(false);
        graph.setAutoCreate(true);
        String css = "edge .notintree {size:1px;fill-color:gray;} " +
                "edge .intree {size:3px;fill-color:black;}";

        graph.addAttribute("ui.stylesheet", css);

        //Add labels
        for (int i = 0; i < labNum; i++) {
            String id = labels[i] + "";
            Node label = graph.addNode(id);
            double occ_num = cluser_max.extractVector(false, i).elementSum();
            label.addAttribute("occurence", occ_num);
        }

        //Add dependency based on prior probability
        for (int r = 0; r < labNum; r++) {
            for (int c = 0; c < labNum; c++) {
                if (r == c) {
                    continue;
                } else {
                    Node node1 = graph.getNode(labels[r] + "");
                    Node node2 = graph.getNode(labels[c] + "");
                    if (node1.hasEdgeBetween(node2)) {
                        continue;
                    } else {
                        double coocurrenceP = co_max.get(r, c) / intNum;
                        if(coocurrenceP > 0) {
                        double occ1P = (double) node1.getAttribute("occurence") / intNum;
                        double occ2P = (double) node2.getAttribute("occurence") / intNum;
                        double dependency = coocurrenceP * (coocurrenceP / (occ1P * occ2P));
//need to review depenency calculation.

//                    double total_occ = node1.getAttribute("occurence");
//                    double dependency = coocurrence/total_occ;
//                    boolean has_depend =  node1.hasEdgeBetween(node2);
//
//                    if(dependency > 0.1) {
//                        if(has_depend ){
//                            Edge exist_dep = node1.getEdgeBetween(node2);
//                            if ((double)exist_dep.getAttribute("weight") > dependency){
//                                continue;
//                            }else {
//                                graph.removeEdge(exist_dep);
//                            }
//                        }


                            //System.out.println(dependency);
                            Edge dependence = graph.addEdge(node1.getId() + "_" + node2.getId(), node1, node2, false);
                            dependence.setAttribute("weight", dependency);
                        }
                    }
                }
            }
        }



            Prim prim = new Prim("weight","ui.class", "intree", "notintree");
            prim.init(graph);
            prim.compute();



            //Reconstructing tree graph
            Graph tree = new SingleGraph("tree");
            tree.setStrict(false);
            tree.setAutoCreate(true);
            Iterator<Node> nodes = graph.getNodeIterator();
            Iterator<Edge> tree_edges =prim.getTreeEdgesIterator();

        while (nodes.hasNext()){
                Node n = nodes.next();
                Node tree_n = tree.addNode(n.getId());
                tree_n.setAttribute("occurence", (double)n.getAttribute("occurence"));
            }

        while (tree_edges.hasNext()){
            Edge tree_e = tree_edges.next();
            Edge new_e = tree.addEdge(tree_e.getId(), tree_e.getNode0().getId(), tree_e.getNode1().getId(), false);
            new_e.setAttribute("weight", (double)tree_e.getAttribute("weight"));
            }

        prim.clear();
        //graph.clear();

        //Calculate shortest path pair
        APSP apsp = new APSP();
        apsp.init(tree);
        apsp.setDirected(false);
        apsp.setWeightAttributeName("weight");
        apsp.compute();

        //calculate centroid
        Centroid centroid = new Centroid();
        centroid.init(tree);
        centroid.compute();

        //initialise root node.
        Node root= tree.getNode(0);
        ArrayList<Node> nodeSet = new ArrayList<>();

        //Find cnetroid root
        for(Node n: tree.getEachNode()){
            nodeSet.add(n);
            if ((boolean)n.getAttribute("centroid")){
                root = n;
                System.out.println("root is "+n.getId());
            }
        }

        // Calculate distance to roo
        for(Node n: nodeSet){
            if (n == root){
                n.setAttribute("dis", 0.0);
            }else {
                APSP.APSPInfo n1Info = n.getAttribute(APSP.APSPInfo.ATTRIBUTE_NAME);
                n.setAttribute("dis", n1Info.getLengthTo(root.getId()));
            }
        }

        //Ranking
        Collections.sort(nodeSet, Comparator.comparing(s -> s.getAttribute("dis")));

        //Pack to array
       for(int i =0; i < labNum; i++){
           int ind = Integer.parseInt(nodeSet.get(i).getId());
           finalisedOrder[i] = ind;
       }
        System.out.println(Arrays.toString(finalisedOrder));

        graph.display();
        tree.display();
//        Base_CC cc = new Base_CC();
//
//        System.out.println(cluster);
//        System.out.println(finalisedOrder.length);
//        cc.prepareChain(finalisedOrder);
//        MLUtils.prepareData(cluster);
//        cc.buildClassifier(cluster);
//
//        String top = "PCut1";
//        String vop = "3";
//        Result r = Evaluation.cvModel(cc, cluster,10,top, vop );
//        System.out.println(Arrays.toString(finalisedOrder));
        //System.out.println(r);
        }
    }

