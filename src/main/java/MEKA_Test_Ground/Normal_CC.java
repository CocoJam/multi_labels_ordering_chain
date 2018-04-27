package MEKA_Test_Ground;

import meka.classifiers.multilabel.BR;
import meka.core.MLUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Normal_CC {
    public static void main(String[] args) throws IOException {
        BufferedReader reader = null;

        try {

            BufferedReader reader = new BufferedReader(
                    new FileReader("src/main/mediamill.arff"));
            Instances data = new Instances(reader);

            MLUtils.prepareData(data);

            System.out.println("Build BR classifier");
            BR classifier = new BR();
            // further configuration of classifier
            classifier.buildClassifier(data);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        finally {
            reader.close();
        }
        //This is the method to read the file and parse everything in to instances

    }
}
