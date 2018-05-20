package Prepare_CC;

import io.jenetics.*;
import io.jenetics.engine.*;
import io.jenetics.util.*;
import meka.core.Result;
import mst.In;
import scala.Int;
import weka.core.Instances;

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
import java.util.stream.IntStream;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static java.util.Objects.requireNonNull;


public class GA_CC extends Thread implements Problem<ISeq<Integer>, EnumGene<Integer>, Double> {
    private final ISeq<Integer> _points;
    public final Cluster_CC_Builder cluster_cc_builder;
    public final Thread thread;
    public String trackingString = "";
    public Result result;
    public int[] trainedChain;

    public EvolutionStatistics<Double, ?> statistics;
//    public File file;
    public  BufferedWriter bufferedWriter;
//    public static GA_CC of(Instances data, double threadhold, int iteration, int popSize) throws Exception {
//        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(data, threadhold);
//        return of(cluster_cc_builder, iteration, popSize);
//    }

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

    @Override
    public Function<ISeq<Integer>, Double> fitness() {
        return p ->
        {
            int[] q = Arrays.stream(p.toArray(new Integer[p.size()])).mapToInt(Integer::intValue).toArray();
            try {
//                System.out.println(Arrays.toString(q));

                result = (CC_Util.ccRun(cluster_cc_builder, 66, q));
                double hamming_loss = Double.parseDouble(result.getMeasurement("Hamming score").toString());
                double exact_match = Double.parseDouble(result.getMeasurement("Exact match").toString());
                double accuracy = Double.parseDouble(result.getMeasurement("Accuracy").toString());
                double averaging = ((1 - hamming_loss) + exact_match + accuracy) / 3;
                trackingString+=( "Accuracy averages: " + averaging);
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
        Engine<EnumGene<Integer>, Double> engine = Engine.builder(basic_ga).mapping(EvolutionResult.toUniquePopulation()).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>(2)).offspringSelector(new TournamentSelector<>(2)).alterers(new SwapMutator<>(0.25), new PartiallyMatchedCrossover<>(1)).build();
        statistics = EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>, Double> best = engine.stream().limit(iterations).peek(r ->
                {
                        trackingString+=("Generate,"+r.getTotalGenerations() + ",Best genotype fitness," + r.getBestFitness()+"\n");
                }
        ).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
        int[] blah = new int[enumGene.length()];
        for (int i = 0; i < enumGene.length(); i++) {
            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
        }
        System.out.println(Arrays.toString(blah));
        trackingString +=("Best Genotype label chains: "+Arrays.toString(blah)+"\n");

        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
        trackingString +=( "overall Stats: \n");
        trackingString +=(statistics.toString()+"\n");
        trackingString +=("Population size: "+engine.getPopulationSize()+",Alterers: "+engine.getAlterer()+ ",SurvivorsSelectors: "+engine.getSurvivorsSelector()+",Optimizer: "+engine.getOptimize()+"\n");
        trackingString +=("Total time "+time2);
//        trackingString+="Total time: "+time2;
        System.out.println(time2);
        bufferedWriter = new BufferedWriter(new FileWriter(new File("cluster_"+cluster_cc_builder.clusterNum+"Information.csv")));
        bufferedWriter.write(trackingString);
        bufferedWriter.close();
        return blah;
    }

    private int[] getOptimalChainEx(Cluster_CC_Builder cluster_cc_builder, int popSize, int iterations, int threads) throws IOException {
        long time1 = System.nanoTime();
        ExecutorService executorService = Executors.newFixedThreadPool(threads);
        GA_CC basic_ga = GA_CC.of(cluster_cc_builder, iterations, popSize);
        Engine<EnumGene<Integer>, Double> engine = Engine.builder(basic_ga).mapping(EvolutionResult.toUniquePopulation()).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>(2)).offspringSelector(new TournamentSelector<>(2)).alterers(new SwapMutator<>(0.25), new PartiallyMatchedCrossover<>(1)).executor(executorService).build();
        statistics = EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>, Double> best = engine.stream().limit(iterations).peek(r ->
                {
                       trackingString +=("Generate,"+r.getTotalGenerations() + ",Best genotype fitness," + r.getBestFitness()+"\n");

                }
        ).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
        int[] blah = new int[enumGene.length()];
        for (int i = 0; i < enumGene.length(); i++) {
            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
        }
        System.out.println(Arrays.toString(blah));
        bufferedWriter.write("Best Genotype label chains: "+Arrays.toString(blah)+"\n");

        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
        trackingString +=( "overall Stats: \n");
        trackingString +=(statistics.toString()+"\n");
        trackingString +=("Population size: "+engine.getPopulationSize()+",Alterers: "+engine.getAlterer()+ ",SurvivorsSelectors: "+engine.getSurvivorsSelector()+",Optimizer: "+engine.getOptimize()+"\n");
        trackingString +=("Total time "+time2);
//        trackingString+="Total time: "+time2;
        System.out.println(time2);
        bufferedWriter = new BufferedWriter(new FileWriter(new File("cluster_"+cluster_cc_builder.clusterNum+"Information.csv")));
        bufferedWriter.write(trackingString);
        bufferedWriter.close();
        return blah;
    }


    public static void main(String[] args) throws Exception {
        long time1 = System.nanoTime();
        List<GA_CC> ga_ccs = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff", i, 0);
            GA_CC ga_cc = GA_CC.of(cluster_cc_builder, 20, 10);
            ga_cc.thread.start();
            ga_ccs.add(ga_cc);
        }

        for (GA_CC ga_cc : ga_ccs) {
            ga_cc.thread.join();
        }
        System.out.println("OverallTime: ");
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime() - time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);

    }
}
