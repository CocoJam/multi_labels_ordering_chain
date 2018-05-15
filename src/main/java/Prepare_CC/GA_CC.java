package Prepare_CC;

import io.jenetics.*;
import io.jenetics.engine.*;
import io.jenetics.util.*;
import mst.In;
import scala.Int;
import weka.core.Instances;

import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static java.util.Objects.requireNonNull;


public class GA_CC implements Problem<ISeq<Integer>, EnumGene<Integer>, Integer> {
    private final ISeq<Integer> _points;
    private final Cluster_CC_Builder cluster_cc_builder;
    public static GA_CC of(Instances data, double threadhold) throws Exception {
        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder(data,threadhold);
        return of(cluster_cc_builder);
    }
    public static GA_CC of( Cluster_CC_Builder cluster_cc_builder) {
        final MSeq<Integer> points = MSeq.ofLength(cluster_cc_builder.labelChain.length);
        for (int i = 0; i < cluster_cc_builder.labelChain.length; ++i) {
            points.set(i,cluster_cc_builder.labelChain[i]);
        }
        return new GA_CC(points.toISeq(), cluster_cc_builder);
    }

    public GA_CC(ISeq<Integer> _points, Cluster_CC_Builder cluster_cc_builder) {
        this._points =requireNonNull(_points);
        this.cluster_cc_builder = cluster_cc_builder;
    }

    @Override
    public Function<ISeq<Integer>, Integer> fitness() {
        return p->
        {
            System.out.println(p);
            //Need to run eval here will do later
            return IntStream.range(0, p.length()).sum();
        };
    }

    @Override
    public Codec<ISeq<Integer>, EnumGene<Integer>> codec() {
        return Codecs.ofPermutation(_points);
    }

    public static void main(String[] args) throws Exception {

        Cluster_CC_Builder cluster_cc_builder = new Cluster_CC_Builder("src/main/CAL500_clustered_adjusted.arff",0,0);
        GA_CC basic_ga = GA_CC.of(cluster_cc_builder);
        Engine<EnumGene<Integer>, Integer> engine  = Engine.builder(basic_ga).optimize(Optimize.MAXIMUM).populationSize(10).alterers(new SwapMutator<>(),new PartiallyMatchedCrossover<>(0.35)).build();
        EvolutionStatistics<Integer,?> statistics =  EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>,Integer> best = engine.stream().limit(250).peek(r -> System.out.println(r.getTotalGenerations() + ": " + r.getGenotypes())).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        System.out.println(best.getGenotype().getChromosome().getGene());
    }
}
