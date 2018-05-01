package GA_Test_Ground;

import io.jenetics.*;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.Factory;
import io.jenetics.util.Seq;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;


public class Basic_GA {

    public static int eval(final Genotype<IntegerGene> genotype){
        return summationOfGene(genotype);
    }

    private static int summationOfGene (Genotype<IntegerGene> genotype){
       int sum = 0;
        for (Chromosome<IntegerGene> integerGenes : genotype) {
            for (IntegerGene integerGene : integerGenes) {
               sum+= integerGene.getAllele();
            }
        }
        return  sum;
    }
    public static void main(String[] args) {

        Factory< Genotype<IntegerGene> > factory = Genotype.of(
                IntegerChromosome.of(1,10, 8),
                IntegerChromosome.of(5,20,5),
                IntegerChromosome.of(3,6,10),
                IntegerChromosome.of(1,5,20),
                IntegerChromosome.of(5,5,5));
        final Engine<IntegerGene, Integer> engine = Engine.builder(Basic_GA::eval, factory).offspringFraction(0.7)
                .survivorsSelector(new RouletteWheelSelector<>()).populationSize(500).alterers(new Mutator<>(0.05),new SinglePointCrossover<>(0.125))
                .offspringSelector(new TournamentSelector<>())
                .build();

        EvolutionStatistics<Integer, ?> statistics = EvolutionStatistics.ofNumber();
        final EvolutionResult<IntegerGene, Integer> results = engine.stream().limit(1000)
                .peek(r -> System.out.println(r.getTotalGenerations() + ": " + r.getGenotypes()))
                .peek(statistics)
                .collect(toBestEvolutionResult());


        Seq<Phenotype<IntegerGene, Integer>> a1 = results.getPopulation();

        System.out.println(a1);
        System.out.println(statistics);
        System.out.println(statistics.getAltered());

        System.out.println(engine.getPopulationSize());
        System.out.println(engine.getAlterer());
        System.out.println(engine.getFitnessFunction().toString());

        System.out.println(results);
    }
}
