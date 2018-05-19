package Prepare_CC;

import meka.classifiers.multilabel.CC;
import meka.classifiers.multilabel.cc.CNode;
import meka.core.A;
import weka.core.Instances;

import java.util.Arrays;

public class Base_CC extends CC{

    @Override
    public void buildClassifier(Instances D) throws Exception {
        this.testCapabilities(D);

        if (this.getDebug()) {
            System.out.print(":- Chain (");
        }

        this.nodes = new CNode[this.m_Chain.length];
        int[] pa = new int[0];
        int[] var4 = this.m_Chain;
        int var5 = var4.length;

        for(int var6 = 0; var6 < var5; ++var6) {
            int j = var4[var6];
            if (this.getDebug()) {
                System.out.print(" " + D.attribute(j).name());
            }
//            System.out.println("j: "+j);
//            System.out.println("pa: "+ Arrays.toString(pa));
            this.nodes[j] = new CNode(j, (int[])null, pa);
            this.nodes[j].build(D, this.m_Classifier);
            pa = A.append(pa, j);
        }

        if (this.getDebug()) {
            System.out.println(" ) -:");
        }

        this.confidences = new double[this.m_Chain.length];
    }
}
