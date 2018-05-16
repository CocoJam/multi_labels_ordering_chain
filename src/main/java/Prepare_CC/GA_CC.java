package Prepare_CC;

import io.jenetics.*;
import io.jenetics.engine.*;
import io.jenetics.util.*;
import mst.In;
import scala.Int;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static java.util.Objects.requireNonNull;


public class GA_CC extends Thread implements Problem<ISeq<Integer>, EnumGene<Integer>, Double> {
    private final ISeq<Integer> _points;
    private final Cluster_CC_Builder cluster_cc_builder;
    public final Thread thread;
    public final String trackingString = "";
    public static GA_CC of(Instances data, double threadhold, int iteration, int popSize) throws Exception {
        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(data,threadhold);
        return of(cluster_cc_builder, iteration, popSize);
    }

    public static GA_CC of( Cluster_CC_Builder cluster_cc_builder, int iteration, int popSize) {
        final MSeq<Integer> points = MSeq.ofLength(cluster_cc_builder.sqeuenceChain.length);
        for (int i = 0; i < cluster_cc_builder.labelChain.length; ++i) {
            points.set(i,cluster_cc_builder.sqeuenceChain[i]);
        }
        return new GA_CC(points.toISeq(), cluster_cc_builder , iteration, popSize);
    }

    public GA_CC(ISeq<Integer> _points, Cluster_CC_Builder cluster_cc_builder, int iteration, int popSize) {
        this._points =requireNonNull(_points);
        this.cluster_cc_builder = cluster_cc_builder;
        this.thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    getOptimalChain(cluster_cc_builder,popSize,iteration);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
//        this.thread.start();
    }

    @Override
    public Function<ISeq<Integer>, Double> fitness() {
        return p->
        {
            int[] q = Arrays.stream(p.toArray(new Integer[p.size()])).mapToInt(Integer::intValue).toArray();
            try {
               return (CC_Util.ccRun(cluster_cc_builder,66,q)*100);
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

    private int[] getOptimalChain( Cluster_CC_Builder cluster_cc_builder,int popSize, int iterations) {
        long time1= System.nanoTime();
        GA_CC basic_ga = GA_CC.of(cluster_cc_builder,iterations,popSize);
        Engine<EnumGene<Integer>, Double> engine  = Engine.builder(basic_ga).optimize(Optimize.MAXIMUM).populationSize(popSize).survivorsSelector(new EliteSelector<>()).alterers(new SwapMutator<>(),new PartiallyMatchedCrossover<>(0.35)).build();
        EvolutionStatistics<Double,?> statistics =  EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>,Double> best = engine.stream().limit(iterations).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        Chromosome<EnumGene<Integer>> enumGene = best.getGenotype().getChromosome();
        int[] blah = new int[enumGene.length()];
        for (int i = 0; i < enumGene.length(); i++) {
            blah[i] = Integer.parseInt(enumGene.getGene(i).toString());
        }
        System.out.println(Arrays.toString(blah));
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
        return blah;
    }

    public static void main(String[] args) throws Exception {
        long time1= System.nanoTime();
        List<GA_CC> ga_ccs = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff",0,0);
            GA_CC ga_cc = GA_CC.of(cluster_cc_builder,100,10);
            ga_cc.thread.start();
            ga_ccs.add(ga_cc);
        }

        for (GA_CC ga_cc : ga_ccs) {
            ga_cc.thread.join();
        }
        System.out.println("OverallTime: ");
        long time2 = TimeUnit.SECONDS.convert(System.nanoTime()-time1, TimeUnit.NANOSECONDS);
        System.out.println(time2);
    }
}
