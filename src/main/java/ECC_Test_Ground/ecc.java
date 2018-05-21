package ECC_Test_Ground;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.io.PrintWriter;

import Prepare_CC.Cluster_CC_Builder;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;


import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.Result;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.classifiers.functions.SMO;

import Prepare_CC.CC_Util;
import WEKA_Test_Ground.Cluster_Fliter;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLUtils;
import meka.core.Result;
import meka.classifiers.multilabel.meta.BaggingML;



public class ecc {

    //        for i in range(10)]

    public static BaggingML EccTest( String file, String fileNum) throws Exception {
        // 8 clusters
//        for (int i = 0; i < 8; i++) {
//            Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff",0,0);
//
//        }

//        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
//        Instances data = source.getDataSet();


//        data = CC_Util.filter(data,clusterNum);
//        CC cc = new CC();
//        MLUtils.prepareData(data);
//        cc.buildClassifier(data);
//        String top = "PCut1";
//        String vop = "3";
//        int numOfCV = data.numInstances()>10? 10:data.numInstances();
//        Result result = Evaluation.cvModel(cc, data, numOfCV, top, vop);
//        System.out.println(result);
//        System.out.println(Arrays.toString(cc.retrieveChain()));


//        Instances D = DataSource.read("src/test/resources/" + fn);
//        MLUtils.prepareData(data);
//        return D;
//
//        Load Music
//        Instances D = loadInstances("Music.arff");
//        Instances D_train = new Instances(data,0,400);
//        Instances D_test = new Instances(data,400,data.numInstances()-400);
//         Train ECC
//        A Bagging ensemble of I chains, bagging P % of the instances
//        default I = 10 P = 1
//        BaggingML h = new BaggingML();
//        h.buildClassifier(data);
//        h.setNumIterations(100);
//        h.setBagSizePercent(100);
//        CC cc = new CC();
//        cc.setClassifier();
//        h.setClassifier(cc);

        // Eval
        BaggingML h = new BaggingML();
        h.globalInfo();
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(file);
        Instances dataset = source.getDataSet();
        Cluster_CC_Builder cluster_cc_builder= new Cluster_CC_Builder(0,dataset,0.0);
        Instances data = cluster_cc_builder.parsedCluster;
        MLUtils.prepareData(data);
        Result result = Evaluation.cvModel(h,data,10,"PCut1","3");

        //Evaluation metrics
        double hamming_loss = Double.parseDouble(result.getMeasurement("Hamming score").toString());
        double exact_match = Double.parseDouble(result.getMeasurement("Exact match").toString());
        double accuracy = Double.parseDouble(result.getMeasurement("Accuracy").toString());
        double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;

        System.out.println(result);
        System.out.println("== Additional Measurements (Evaluation Metrics Averaging)");
        System.out.println("averaging:" + averaging);
        PrintWriter writer = new PrintWriter("result"+ fileNum + ".txt", "UTF-8");
        writer.println(result);
        writer.println("== Additional Measurements (Evaluation Metrics)");
        writer.println("hamming_loss:" + hamming_loss);
        writer.println("exact_match:" + exact_match);
        writer.println("accuracy:" + accuracy);
        writer.println("averaging:" + averaging);
        writer.close();

//        BaggingML h = new BaggingML();
//        CC cc = new CC();
//        cc.setClassifier(new SMO());
//        h.setClassifier(cc);
//        Result result = Evaluation.cvModel(cc, data, numOfCV, top, vop);
        return h;
    }

    public static void main(String[] args) throws Exception {
//        ECC trains m CC classifiers C1, C2, · · · , Cm. Each Ck is trained with:
//        – a random chain ordering (of L); and
//        – a random subset of D

        //Meka command line
        //http://waikato.github.io/meka/meka.classifiers.multilabel.meta.BaggingML/

        //A Bagging ensemble of I chains, bagging P % of the instances
        //java meka.classifiers.multilabel.meta.BaggingML -I 10 -P 100 -t data.arff -W meka.classifiers.multilabel.CC -- -W weka.classifiers.functions.SMO
        //I Sets the number of models
        //-p Size of each bag, as a percentage of total training size (default 67)
        //-W <classifier name>
        //Full name of base classifier. (default: meka.classifiers.multilabel.CC)
        for (int i=0;i<10;i++){
            ecc.EccTest("src/main/CAL500_clustered_adjusted.arff", Integer.toString(i));
        }
    }
}
