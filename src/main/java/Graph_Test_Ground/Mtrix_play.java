package Graph_Test_Ground;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by Administer on 19/05/2018.
 */
public class Mtrix_play {


    public static void main(String[] args) throws Exception {

        double[] input = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        SimpleMatrix trial = new SimpleMatrix(4, 2, true, input);
        SimpleMatrix firstV = trial.extractVector(false, 0);
        SimpleMatrix secV = trial.extractVector(false, 1);
        SimpleMatrix multi = firstV.elementMult(secV);
        System.out.println(trial);
        System.out.println(firstV);
        System.out.println(secV);
        System.out.println(multi);

    }

}
