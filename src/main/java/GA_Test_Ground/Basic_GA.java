package GA_Test_Ground;

import io.jenetics.*;
import io.jenetics.engine.*;
import io.jenetics.util.*;
import mst.In;
import scala.Int;

import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;
import static java.util.Objects.requireNonNull;


public class Basic_GA implements Problem<ISeq<Integer>, EnumGene<Integer>, Integer> {
    private final ISeq<Integer> _points;

    public static Basic_GA of( int[] range) {
        final MSeq<Integer> points = MSeq.ofLength(range.length);
        for (int i = 0; i < range.length; ++i) {
            points.set(i,range[i]);
        }
        return new Basic_GA(points.toISeq());
    }

    public Basic_GA(ISeq<Integer> _points) {
        this._points =requireNonNull(_points);
    }

    public static int eval(final Genotype<IntegerGene> genotype) {
        return summationOfGene(genotype);
    }

    private static int summationOfGene(Genotype<IntegerGene> genotype) {
        int sum = 0;
        for (Chromosome<IntegerGene> integerGenes : genotype) {
            for (IntegerGene integerGene : integerGenes) {
                sum += integerGene.getAllele();
            }
        }
        return sum;
    }

    public static void main(String[] args) {

        int [] ints = new int[]{1,3,2,5,6,7,8,10,9};
        Basic_GA basic_ga = Basic_GA.of( ints);
        Engine<EnumGene<Integer>, Integer> engine  = Engine.builder(basic_ga).optimize(Optimize.MAXIMUM).populationSize(10).alterers(new SwapMutator<>(),new PartiallyMatchedCrossover<>(0.35)).build();
        EvolutionStatistics<Integer,?> statistics =  EvolutionStatistics.ofNumber();
        Phenotype<EnumGene<Integer>,Integer> best = engine.stream().limit(250).peek(r -> System.out.println(r.getTotalGenerations() + ": " + r.getGenotypes())).peek(statistics).collect(toBestPhenotype());
        System.out.println(statistics);
        System.out.println(best.getGenotype().getChromosome().getGene());
    }

    @Override
    public Function<ISeq<Integer>, Integer> fitness() {
        return p-> IntStream.range(0, p.length()).sum();
    }

    @Override
    public Codec<ISeq<Integer>, EnumGene<Integer>> codec() {
        return Codecs.ofPermutation(_points);
    }
}
