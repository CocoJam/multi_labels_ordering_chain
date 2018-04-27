package GA_Test_Ground;

import io.jenetics.*;
import io.jenetics.internal.math.probability;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;

import java.util.Random;

public class CustomAlter<G extends Gene<?, G>, C extends Comparable<? super C>>extends AbstractAlterer<G, C>{

    protected CustomAlter(double probability) {
        super(probability);
    }

    public AltererResult<G, C> alter(Seq<Phenotype<G, C>> population, long l) {
        assert population != null : "Not null is guaranteed from base class.";

//        Random random = RandomRegistry.getRandom();
//        double p = Math.pow(this._probability, 0.3333333333333333D);
//        int P = probability.toInt(p);
//        Seq<MutatorResult<Phenotype<G, C>>> result = population.map((pt) -> {
//            return random.nextInt() < P ? this.mutate(pt, generation, p, random) : MutatorResult.of(pt);
//        });
//        return AltererResult.of(result.map(MutatorResult::getResult).asISeq(), result.stream().mapToInt(MutatorResult::getMutations).sum());
        return null;
    }

    public static void main(String[] args) {
//        Seq<Phenotype<IntegerGene, Integer>> a1 =  Genotype.of(
//                IntegerChromosome.of(1,10, 8),
//                IntegerChromosome.of(5,20,5),
//                IntegerChromosome.of(3,6,10),
//                IntegerChromosome.of(1,5,20),
//                IntegerChromosome.of(5,5,5));
//        System.out.println(a1.length());
    }

}
