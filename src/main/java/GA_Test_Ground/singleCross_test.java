package GA_Test_Ground;

import io.jenetics.*;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.EvolutionStatistics;
import io.jenetics.util.Factory;

import static io.jenetics.engine.EvolutionResult.toBestEvolutionResult;
import static io.jenetics.engine.EvolutionResult.toBestPhenotype;

public class singleCross_test {

    public static void main(String[] args) {
        Factory< Genotype<IntegerGene> > factory = Genotype.of(new IntegerChromosome (100,150,3),
               new IntegerChromosome (10,20,5)
                );

        final Engine<IntegerGene, Integer> engine = Engine.builder(Basic_GA::eval, factory)
                .survivorsSelector(new RouletteWheelSelector<>()).populationSize(10).alterers(new Mutator<>(0.05),new SinglePointCrossover<>(0.0125))
                .offspringSelector(new TournamentSelector<>())
                .build();


        EvolutionStatistics<Integer, ?> statistics = EvolutionStatistics.ofNumber();

//        final Phenotype<IntegerGene, Integer> results = engine.stream().limit(2)
//                .peek(r -> System.out.println(r.getTotalGenerations() + ": " + r.getGenotypes()))
////                .peek(statistics)
//                .collect(toBestPhenotype());
        System.out.println(engine.getAlterer().toString());
        if(engine.getAlterer() instanceof SinglePointCrossover){
            System.out.println("Asd");
        }
//        System.out.println(results);
    }
}
