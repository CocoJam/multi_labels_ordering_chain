package WEKA_Test_Ground;

import Prepare_CC.Cluster_CC_Builder;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class BaggingMLEvaluation extends Thread {
    private enum TYPE {NO_CLUSTER, CLUSTER_DROP_LABEL,CLUSTER_LABEL}

    public int w = 0;
    public TYPE choice = null;

    public BaggingMLEvaluation(int fileNum, TYPE num) {
        this.w = fileNum;
        this.choice = num;
    }


    public void ECC_Cluster_Drop_Labels(){
        String Tracking = "Sample,Hamming_loss,Exact_match,Accuracy,\n";
        int sampleNumber = 1;
        List<Double> ham = new ArrayList<>();
        List<Double> exact = new ArrayList<>();
        List<Double> acc = new ArrayList<>();
        Instances train = null;
        Instances test = null;
        for (int i = 0; i < 10; i++) {
            try {
                train = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
                test = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
            } catch (Exception e) {
                e.printStackTrace();
            }
            int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount;
//        System.out.println(train.attributeStats(train.numAttributes()-1).distinctCount);
            List<Cluster_CC_Builder> cluster_cc_builders = new ArrayList<>();

            for (int j = 0; j < numberOfCluster; j++) {
                Cluster_CC_Builder cluster_cc_builder = null;
                try {
                    cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                cluster_cc_builders.add(cluster_cc_builder);
            }

            Instances testInstances = null;
            try {
                testInstances = Cluster_Fliter.knn_inference(train, test, 3);
            } catch (Exception e) {
                e.printStackTrace();
            }
//        Instances train = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_train.arff")).getDataSet();
            for (int j = 0; j < cluster_cc_builders.size(); j++) {
//                cluster_cc_builders.get(j).parsedCluster.deleteAttributeAt(cluster_cc_builders.get(j).parsedCluster.numAttributes()-1);
                cluster_cc_builders.get(j).parsedCluster.deleteAttributeAt(cluster_cc_builders.get(j).parsedCluster.numAttributes() - 1);

                Instances clusterX = Cluster_Fliter.filter(testInstances, j);
                clusterX.deleteAttributeAt(clusterX.numAttributes() - 1);
                Remove remove = new Remove();
                remove.setAttributeIndicesArray(cluster_cc_builders.get(j).labelsDropped);

                try {
                    remove.setInputFormat(clusterX);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                try {
                    clusterX = Filter.useFilter(clusterX, remove);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                Pattern pattern = Pattern.compile("(.+C )(\\d+)");
                Matcher matcher = pattern.matcher(clusterX.relationName());
                if (matcher.find()) {
                    System.out.println(matcher.group(0));
                    int numLabels = Integer.parseInt(matcher.group(2));
                    clusterX.setRelationName(matcher.group(1) + (numLabels - (cluster_cc_builders.get(j).labelsDropped.length)));
                    System.out.println(clusterX.relationName());
                }
                try {
                    MLUtils.prepareData(cluster_cc_builders.get(j).parsedCluster);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                try {
                    MLUtils.prepareData(clusterX);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                BaggingML baggingML = new BaggingML();
                String top = "PCut1";
                String vop = "3";
                try {
                    baggingML.buildClassifier(cluster_cc_builders.get(j).parsedCluster);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                Result evaluateModel = null;
                try {
                    evaluateModel = Evaluation.evaluateModel(baggingML, cluster_cc_builders.get(j).parsedCluster, clusterX, top, vop);
                } catch (ArrayIndexOutOfBoundsException e) {
                    System.out.println(e);
                    continue;
                } catch (Exception e) {
                    e.printStackTrace();
                }
//                    System.out.println(evaluateModel);
                Tracking += sampleNumber + ",";
                sampleNumber++;
                double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                ham.add(hamming_loss);
                Tracking += hamming_loss + ",";
                double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                exact.add(exact_match);
                Tracking += exact_match + ",";
                double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
                acc.add(accuracy);
                Tracking += accuracy + ",\n";
            }
        }
        double ham_summ = ham.stream().reduce(0.0, Double::sum);
        double exact_summ = exact.stream().reduce(0.0, Double::sum);
        double acc_summ = acc.stream().reduce(0.0, Double::sum);

        double ham_average = ham_summ / sampleNumber;
        double exact_average = exact_summ / sampleNumber;
        double acc_average = acc_summ / sampleNumber;

        double ham_var = ham.stream().reduce(0.0, (x, y) -> x + Math.pow((y - ham_average), 2));
        double exact_var = exact.stream().reduce(0.0, (x, y) -> x + Math.pow((y - exact_average), 2));
        double acc_var = acc.stream().reduce(0.0, (x, y) -> x + Math.pow((y - acc_average), 2));
        Tracking += "Average," + ham_average + "," + exact_average + "," + acc_average + ",\n";
        Tracking += "varience," + ham_var / sampleNumber + "," + exact_var / sampleNumber + "," + acc_var / sampleNumber + ",\n";
        Tracking += "standard deviation," + Math.sqrt(ham_var / sampleNumber) + "," + Math.sqrt(exact_var / sampleNumber) + "," + Math.sqrt(acc_var / sampleNumber) + ",\n";
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(new File("CVseed_" + w + "/LABEL_DROP_ECC.csv")));
            bufferedWriter.write(Tracking);
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("One trial: ");
    }


    public void ECC_Cluster_NON_DROP_LABELS(){
        String Tracking = "Sample,Hamming_loss,Exact_match,Accuracy,\n";
        int sampleNumber = 1;
        List<Double> ham = new ArrayList<>();
        List<Double> exact = new ArrayList<>();
        List<Double> acc = new ArrayList<>();
        Instances train = null;
        Instances test = null;
        for (int i = 0; i < 10; i++) {
            try {
                train = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
                test = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
            } catch (Exception e) {
                e.printStackTrace();
            }
            int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount;
//        System.out.println(train.attributeStats(train.numAttributes()-1).distinctCount);
            List<Cluster_CC_Builder> cluster_cc_builders = new ArrayList<>();

            for (int j = 0; j < numberOfCluster; j++) {
                Cluster_CC_Builder cluster_cc_builder = null;
                try {
                    cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                cluster_cc_builders.add(cluster_cc_builder);
            }

            Instances testInstances = null;
            try {
                testInstances = Cluster_Fliter.knn_inference(train, test, 3);
            } catch (Exception e) {
                e.printStackTrace();
            }
//        Instances train = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_train.arff")).getDataSet();
            for (int j = 0; j < cluster_cc_builders.size(); j++) {
//                cluster_cc_builders.get(j).parsedCluster.deleteAttributeAt(cluster_cc_builders.get(j).parsedCluster.numAttributes()-1);
                cluster_cc_builders.get(j).cluster.deleteAttributeAt(cluster_cc_builders.get(j).cluster.numAttributes() - 1);

                Instances clusterX = Cluster_Fliter.filter(testInstances, j);
                clusterX.deleteAttributeAt(clusterX.numAttributes() - 1);

                try {
                    MLUtils.prepareData(cluster_cc_builders.get(j).cluster);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                try {
                    MLUtils.prepareData(clusterX);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                BaggingML baggingML = new BaggingML();
                String top = "PCut1";
                String vop = "3";
                try {
                    baggingML.buildClassifier(cluster_cc_builders.get(j).cluster);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                Result evaluateModel = null;
                try {
                    evaluateModel = Evaluation.evaluateModel(baggingML, cluster_cc_builders.get(j).cluster, clusterX, top, vop);
                } catch (ArrayIndexOutOfBoundsException e) {
                    System.out.println(e);
                    continue;
                } catch (Exception e) {
                    e.printStackTrace();
                }
//                    System.out.println(evaluateModel);
                Tracking += sampleNumber + ",";
                sampleNumber++;
                double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                ham.add(hamming_loss);
                Tracking += hamming_loss + ",";
                double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                exact.add(exact_match);
                Tracking += exact_match + ",";
                double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
                acc.add(accuracy);
                Tracking += accuracy + ",\n";
            }
        }
        double ham_summ = ham.stream().reduce(0.0, Double::sum);
        double exact_summ = exact.stream().reduce(0.0, Double::sum);
        double acc_summ = acc.stream().reduce(0.0, Double::sum);

        double ham_average = ham_summ / sampleNumber;
        double exact_average = exact_summ / sampleNumber;
        double acc_average = acc_summ / sampleNumber;

        double ham_var = ham.stream().reduce(0.0, (x, y) -> x + Math.pow((y - ham_average), 2));
        double exact_var = exact.stream().reduce(0.0, (x, y) -> x + Math.pow((y - exact_average), 2));
        double acc_var = acc.stream().reduce(0.0, (x, y) -> x + Math.pow((y - acc_average), 2));
        Tracking += "Average," + ham_average + "," + exact_average + "," + acc_average + ",\n";
        Tracking += "varience," + ham_var / sampleNumber + "," + exact_var / sampleNumber + "," + acc_var / sampleNumber + ",\n";
        Tracking += "standard deviation," + Math.sqrt(ham_var / sampleNumber) + "," + Math.sqrt(exact_var / sampleNumber) + "," + Math.sqrt(acc_var / sampleNumber) + ",\n";
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(new File("CVseed_" + w + "/NON_LABEL_DROP_ECC.csv")));
            bufferedWriter.write(Tracking);
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("One trial: ");
    }

    public void ECC_Without_Cluster() throws Exception {
        String Tracking = "Sample,Hamming_loss,Exact_match,Accuracy,\n";
        int sampleNumber = 1;
        List<Double> ham = new ArrayList<>();
        List<Double> exact = new ArrayList<>();
        List<Double> acc = new ArrayList<>();
        Instances train = null;
        Instances test = null;
        for (int i = 0; i < 10; i++) {
            try {
                train = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
                test = (new ConverterUtils.DataSource("CVseed_" + w + "/CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
            } catch (Exception e) {
                e.printStackTrace();
            }
            train.deleteAttributeAt(train.numAttributes() - 1);
            MLUtils.prepareData(train);
            MLUtils.prepareData(test);

            String top = "PCut1";
            String vop = "3";
            BaggingML baggingML = new BaggingML();
            Result evaluateModel = null;
            evaluateModel = Evaluation.evaluateModel(baggingML, train, test, top, vop);
            Tracking += sampleNumber + ",";
            sampleNumber++;
            double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
            ham.add(hamming_loss);
            Tracking += hamming_loss + ",";
            double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
            exact.add(exact_match);
            Tracking += exact_match + ",";
            double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
            acc.add(accuracy);
            Tracking += accuracy + ",\n";

        }
        double ham_summ = ham.stream().reduce(0.0, Double::sum);
        double exact_summ = exact.stream().reduce(0.0, Double::sum);
        double acc_summ = acc.stream().reduce(0.0, Double::sum);

        double ham_average = ham_summ / sampleNumber;
        double exact_average = exact_summ / sampleNumber;
        double acc_average = acc_summ / sampleNumber;

        double ham_var = ham.stream().reduce(0.0, (x, y) -> x + Math.pow((y - ham_average), 2));
        double exact_var = exact.stream().reduce(0.0, (x, y) -> x + Math.pow((y - exact_average), 2));
        double acc_var = acc.stream().reduce(0.0, (x, y) -> x + Math.pow((y - acc_average), 2));
        Tracking += "Average," + ham_average + "," + exact_average + "," + acc_average + ",\n";
        Tracking += "varience," + ham_var / sampleNumber + "," + exact_var / sampleNumber + "," + acc_var / sampleNumber + ",\n";
        Tracking += "standard deviation," + Math.sqrt(ham_var / sampleNumber) + "," + Math.sqrt(exact_var / sampleNumber) + "," + Math.sqrt(acc_var / sampleNumber) + ",\n";
        BufferedWriter bufferedWriter = null;
        try {
            bufferedWriter = new BufferedWriter(new FileWriter(new File("CVseed_" + w + "/ECC.csv")));
            bufferedWriter.write(Tracking);
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("One trial: ");
    }

    @Override
    public void run() {
        if (choice == TYPE.NO_CLUSTER){
            try {
                ECC_Without_Cluster();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }else if(choice == TYPE.CLUSTER_LABEL){
            ECC_Cluster_NON_DROP_LABELS();
        }
        else if (choice == TYPE.CLUSTER_DROP_LABEL){
            ECC_Cluster_Drop_Labels();
        }

    }

    public static void main(String[] args) throws Exception {
        List<BaggingMLEvaluation> baggingMLS = new ArrayList<>();
        for (int w = 0; w < 10; w++) {
            BaggingMLEvaluation baggingML = new BaggingMLEvaluation(w, TYPE.NO_CLUSTER);
            baggingML.start();
            baggingMLS.add(baggingML);
        }

        for (BaggingMLEvaluation baggingML : baggingMLS) {
            baggingML.join();
        }

//        Instances train = (new ConverterUtils.DataSource("src/main/CAL500.arff")).getDataSet();
////        Instances train = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_train.arff")).getDataSet();
//
//        train.deleteAttributeAt(train.numAttributes()-1);
////        Instances test = (new ConverterUtils.DataSource("Split_" + 1 + "/CAL500_test.arff")).getDataSet();
////        Base_CC cc = new Base_CC();
//        BaggingML baggingML = new BaggingML();
//        MLUtils.prepareData(train);
////        MLUtils.prepareData(test);
//        String top = "PCut1";
//        String vop = "3";
//        baggingML.buildClassifier(train);
//        Result evaluateModel;
//        evaluateModel = Evaluation.cvModel(baggingML, train, 10 ,top,vop);
//        System.out.println(evaluateModel);

    }
}
