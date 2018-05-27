package Prepare_CC;

import WEKA_Test_Ground.Cluster_Fliter;
import io.jenetics.*;
import io.jenetics.engine.*;
import io.jenetics.util.*;
import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.Evaluation;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import mst.In;
import scala.Int;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.experiment.ResultsPanel;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static java.util.Objects.requireNonNull;


public class GA_CC extends Thread implements Problem<ISeq<Integer>, EnumGene<Integer>, Double> {
    private final ISeq<Integer> _points;
    public Cluster_CC_Builder cluster_cc_builder = null;
    public final Thread thread;
    public String trackingString = "";
    public Result result;
    public int[] trainedChain;
    public Instances instances;

    public EvolutionStatistics<Double, ?> statistics;
    //    public File file;
    public BufferedWriter bufferedWriter;
//    public static GA_CC of(Instances data, double threadhold, int iteration, int popSize) throws Exception {
//        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(data, threadhold);
//        return of(cluster_cc_builder, iteration, popSize);
//    }

    public static GA_CC of(Instances data, int iteration, int popSize) throws IOException {
        Pattern pattern = Pattern.compile("(.+-C (\\d+))");
        Matcher matcher = pattern.matcher(data.relationName());
        int numLabels = 0;
        if (matcher.find()) {
            data.setRelationName(matcher.group(0));
            numLabels = Integer.parseInt(matcher.group(2));
        }
        final MSeq<Integer> points = MSeq.ofLength(numLabels);
        for (int i = 0; i < numLabels; ++i) {
            points.set(i, i);
        }
        return new GA_CC(points.toISeq(), data, iteration, popSize);
    }

    private int[] getOptimalChain(Instances instances, int popSize, int iterations) throws IOException {
        long time1 = System.nanoTime();
        ExecutorService executorService = Executors.newFixedThreadPool(10);
        GA_CC basic_ga = GA_CC.of(instances, iterations, popSize);


        Engine<EnumGene<Integer>, Double> engine = Engine.builder(basic_ga).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>(2)).offspringSelector(new TournamentSelector<>(popSize/4)).alterers(new SwapMutator<>(0.25), new PartiallyMatchedCrossover<>(0.35)).build();


        statistics = EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>, Double> best = engine.stream().limit(iterations).peek(r ->
                {
                    trackingString += ("Generate," + r.getTotalGenerations() + ",Best genotype fitness," + r.getBestFitness() + "\n");
                }
        ).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
        int[] blah = new int[enumGene.length()];
        for (int i = 0; i < enumGene.length(); i++) {
            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
        }
        System.out.println(Arrays.toString(blah));
        trackingString += ("Best Genotype label chains: " + Arrays.toString(blah) + "\n");

        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
        trackingString += ("overall Stats: \n");
        trackingString += (statistics.toString() + "\n");
        trackingString += ("Population size: " + engine.getPopulationSize() + ",Alterers: " + engine.getAlterer() + ",SurvivorsSelectors: " + engine.getSurvivorsSelector() + ",Optimizer: " + engine.getOptimize() + "\n");
        trackingString += ("Total time " + time2 + "\n");
//        trackingString+="Total time: "+time2;
        System.out.println(time2);
        bufferedWriter = new BufferedWriter(new FileWriter(new File("Non_clustered_GA_CC.csv")));
        bufferedWriter.write(trackingString);
        bufferedWriter.close();
        return blah;
    }


    public static GA_CC of(Cluster_CC_Builder cluster_cc_builder, int iteration, int popSize) throws IOException {
        final MSeq<Integer> points = MSeq.ofLength(cluster_cc_builder.sqeuenceChain.length);
        for (int i = 0; i < cluster_cc_builder.labelChain.length; ++i) {
            points.set(i, cluster_cc_builder.sqeuenceChain[i]);
        }
        return new GA_CC(points.toISeq(), cluster_cc_builder, iteration, popSize);
    }

    public GA_CC(ISeq<Integer> _points, Cluster_CC_Builder cluster_cc_builder, int iteration, int popSize) throws IOException {
        this._points = requireNonNull(_points);
        this.cluster_cc_builder = cluster_cc_builder;


        this.thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    trainedChain = getOptimalChain(cluster_cc_builder, popSize, iteration);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
//        this.thread.start();
    }

    public GA_CC(ISeq<Integer> _points, Instances instances, int iteration, int popSize) throws IOException {
        this._points = requireNonNull(_points);
        this.instances = instances;


        this.thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    trainedChain = getOptimalChain(instances, popSize, iteration);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
//        this.thread.start();
    }

    @Override
    public Function<ISeq<Integer>, Double> fitness() {
        return p ->
        {
            int[] q = Arrays.stream(p.toArray(new Integer[p.size()])).mapToInt(Integer::intValue).toArray();
            try {
//                System.out.println(Arrays.toString(q));
                if (this.cluster_cc_builder != null){
                result = (CC_Util.ccRun(cluster_cc_builder, 66, q));}
                else{
                    result = (CC_Util.ccRun(instances, 66, q));
                }


                double hamming_loss = Double.parseDouble(result.getMeasurement("Hamming loss").toString());
                double exact_match = Double.parseDouble(result.getMeasurement("Exact match").toString());
                double accuracy = Double.parseDouble(result.getMeasurement("Accuracy").toString());
                double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;
                trackingString += ("Accuracy averages: " + averaging);
//                this.trackingString +=
                return averaging;
            } catch (Exception e) {
                e.printStackTrace();
                return 0.0;
            }
        };
    }

    @Override
    public Codec<ISeq<Integer>, EnumGene<Integer>> codec() {
        return Codecs.ofPermutation(_points);
    }

    private int[] getOptimalChain(Cluster_CC_Builder cluster_cc_builder, int popSize, int iterations) throws IOException {
        long time1 = System.nanoTime();
        GA_CC basic_ga = GA_CC.of(cluster_cc_builder, iterations, popSize);
        Engine<EnumGene<Integer>, Double> engine = Engine.builder(basic_ga).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>(4)).offspringSelector(new TournamentSelector<>(popSize/4)).alterers(new SwapMutator<>(0.25), new PartiallyMatchedCrossover<>(0.35)).build();
        statistics = EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>, Double> best = engine.stream().limit(iterations).peek(r ->
                {
                    trackingString += ("Generate," + r.getTotalGenerations() + ",Best genotype fitness," + r.getBestFitness() + "\n");
                }
        ).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
        int[] blah = new int[enumGene.length()];
        for (int i = 0; i < enumGene.length(); i++) {
            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
        }
        System.out.println(Arrays.toString(blah));
        trackingString += ("Best Genotype label chains: " + Arrays.toString(blah) + "\n");

        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
        trackingString += ("overall Stats: \n");
        trackingString += (statistics.toString() + "\n");
        trackingString += ("Population size: " + engine.getPopulationSize() + ",Alterers: " + engine.getAlterer() + ",SurvivorsSelectors: " + engine.getSurvivorsSelector() + ",Optimizer: " + engine.getOptimize() + "\n");
        trackingString += ("Total time " + time2);
//        trackingString+="Total time: "+time2;
        System.out.println(time2);
        bufferedWriter = new BufferedWriter(new FileWriter(new File("cluster_" + cluster_cc_builder.clusterNum + "Information.csv")));
        bufferedWriter.write(trackingString);
        bufferedWriter.close();
        return blah;
    }
//
//    private int[] getOptimalChainEx(Cluster_CC_Builder cluster_cc_builder, int popSize, int iterations, int threads) throws IOException {
//        long time1 = System.nanoTime();
//        ExecutorService executorService = Executors.newFixedThreadPool(threads);
//        GA_CC basic_ga = GA_CC.of(cluster_cc_builder, iterations, popSize);
//        Engine<EnumGene<Integer>, Double> engine = Engine.builder(basic_ga).mapping(EvolutionResult.toUniquePopulation()).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>(2)).offspringSelector(new TournamentSelector<>(2)).alterers(new SwapMutator<>(0.25), new PartiallyMatchedCrossover<>(1)).executor(executorService).build();
//        statistics = EvolutionStatistics.ofNumber();
//        Phenotype<EnumGene<Integer>, Double> best = engine.stream().limit(iterations).peek(r ->
//                {
//                       trackingString +=("Generate,"+r.getTotalGenerations() + ",Best genotype fitness," + r.getBestFitness()+"\n");
//
//                }
//        ).peek(statistics).collect(toBestPhenotype());
//        System.out.println(statistics);
//        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
//        int[] blah = new int[enumGene.length()];
//        for (int i = 0; i < enumGene.length(); i++) {
//            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
//        }
//        System.out.println(Arrays.toString(blah));
//        bufferedWriter.write("Best Genotype label chains: "+Arrays.toString(blah)+"\n");
//
//        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
//        trackingString +=( "overall Stats: \n");
//        trackingString +=(statistics.toString()+"\n");
//        trackingString +=("Population size: "+engine.getPopulationSize()+",Alterers: "+engine.getAlterer()+ ",SurvivorsSelectors: "+engine.getSurvivorsSelector()+",Optimizer: "+engine.getOptimize()+"\n");
//        trackingString +=("Total time "+time2);
////        trackingString+="Total time: "+time2;
//        System.out.println(time2);
//        bufferedWriter = new BufferedWriter(new FileWriter(new File("cluster_"+cluster_cc_builder.clusterNum+"Information.csv")));
//        bufferedWriter.write(trackingString);
//        bufferedWriter.close();
//        return blah;
//    }


    public static void main(String[] args) throws Exception {
//        for (int i = 0; i < 10; i++) {
//            Cluster_CC_GA_Wrapper cluster_cc_ga_wrapper = new Cluster_CC_GA_Wrapper("src/main/CAL500_train.arff",i);
//            Instances data = (new ConverterUtils.DataSource("src/main/CAL500_test.arff")).getDataSet();
//            Instances dataTest = new Instances(data);
////        List<int[]> results = cluster_cc_ga_wrapper.ResultsChains(cluster_cc_ga_wrapper.listOfClusterBuilder);
//            Instances testInstances  =  Cluster_Fliter.knn_inference(cluster_cc_ga_wrapper.clustered,dataTest,cluster_cc_ga_wrapper.clusterNumber);
//            System.out.println(testInstances);
//        }
//
//        long time1 = System.nanoTime();
        
        for (int i = 0; i < 10; i++) {
            long time1 = System.nanoTime();
            Instances train = (new ConverterUtils.DataSource("CVSplit_" + i + "/CAL500_train.arff")).getDataSet();
            Instances test = (new ConverterUtils.DataSource("CVSplit_" + i + "/CAL500_test.arff")).getDataSet();
            int numberOfCluster = train.attributeStats(train.numAttributes() - 1).distinctCount - 1;
//        System.out.println(train.attributeStats(train.numAttributes()-1).distinctCount);
            List<Cluster_CC_Builder> cluster_cc_builders = new ArrayList<>();
            String ClusterTracking = "";
            for (int j = 0; j < numberOfCluster; j++) {
                Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(j, train, 0);
                cluster_cc_builders.add(cluster_cc_builder);
            }
            List<GA_CC> results = Cluster_CC_GA_Wrapper.ResultsChains(cluster_cc_builders);
            System.out.println("One GA");
            System.out.println(System.nanoTime()-time1);
            for (int j = 0; j < results.size(); j++) {
                ClusterTracking +="Trial,"+i+ ",Cluster,"+j+",best result chain,"+Arrays.toString(results.get(j).trainedChain)+"\n";
                ClusterTracking += results.get(j).result+"\n";
            }

            String overallExact_match= "";
            String overallHamming_loss= "";
            String overallAccuracy= "";
            String overallAverage= "";
            Instances testInstances = Cluster_Fliter.knn_inference(train, test, 3);
            List<Result> resultsList = new ArrayList<>();
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
//                    clusterX.setRelationName(cluster_cc_builder.cluster.relationName());
//                }
                Base_CC cc = new Base_CC();
                MLUtils.prepareData(cluster_cc_builder.cluster);
                MLUtils.prepareData(clusterX);
                cc.prepareChain(results.get(j).trainedChain);
                cc.buildClassifier(cluster_cc_builder.cluster);
                String top = "PCut1";
                String vop = "3";
                Result evaluateModel;
                try{ evaluateModel = Evaluation.evaluateModel(cc, cluster_cc_builder.cluster, clusterX, top, vop);
                    resultsList.add(evaluateModel);
                }
                catch (ArrayIndexOutOfBoundsException e){
                    System.out.println(e);
                    continue;
                }
                double hamming_loss = Double.parseDouble(evaluateModel.getMeasurement("Hamming loss").toString());
                overallHamming_loss += hamming_loss+",";
                double exact_match = Double.parseDouble(evaluateModel.getMeasurement("Exact match").toString());
                overallExact_match += exact_match +",";
                double accuracy = Double.parseDouble(evaluateModel.getMeasurement("Accuracy").toString());
                overallAccuracy += accuracy+",";

                double averaging = (((1 - hamming_loss) + exact_match + accuracy) / 3);
                overallAverage+= averaging+",";
//                System.out.println(evaluateModel);
            }

           BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File("CVSplit_"+i+"/ CVSPLIT_GA_Results_CV_20iteration_10_Pop_Mut_025_Tournament_popDiv4_Elit_4_K_3.csv")));
            bufferedWriter.write(ClusterTracking);
            bufferedWriter.write("Hamming_loss, "+overallHamming_loss+"\n");
            bufferedWriter.write("Exact_match, "+overallExact_match+"\n");
            bufferedWriter.write("Accuracy, "+overallAccuracy+"\n");
            bufferedWriter.write("Averaging, "+overallAverage);
            bufferedWriter.close();
            System.out.println("One trial: ");
            System.out.println(System.nanoTime()-time1);
//            System.out.println(Arrays.toString(cluster_cc_builder.sqeuenceChain));
        }

//        System.out.println(cluster_cc_builder.labelChain.length);
//        System.out.println(cluster_cc_builder.sqeuenceChain.length);
//        Base_CC cc = new Base_CC();
//        MLUtils.prepareData(cluster_cc_builder.parsedCluster);
//
//        cc.prepareChain(cluster_cc_builder.labelChain);
//        System.out.println(cluster_cc_builder.parsedCluster);

//        System.out.println(testInstances);
    }

}
