package Graph_Test_Ground;

import org.graphstream.algorithm.TarjanStronglyConnectedComponents;
import org.graphstream.graph.implementations.MultiGraph;
import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;

import static javax.swing.text.html.CSS.getAttribute;

/**
 * Created by Administer on 21/05/2018.
 */
public class Tarjon {


    public static void main(String[] args) throws Exception {

        Graph g = new MultiGraph("g");

        String nodes = "abcdefgh";
        String edges = "abbccddccgdhhdhggffgbfefbeea";

        for (int i = 0; i < 8; i++) {
            g.addNode(nodes.substring(i, i + 1));
        }
        for (int i = 0; i < 14; i++) {
            g.addEdge(edges.substring(2 * i, 2 * i + 2),
                    edges.substring(2 * i, 2 * i + 1),
                    edges.substring(2 * i + 1, 2 * i + 2), true);
        }



        TarjanStronglyConnectedComponents tscc = new TarjanStronglyConnectedComponents();
        tscc.init(g);
        tscc.compute();

        String attr = tscc.getSCCIndexAttribute();
        for (Node n : g.getEachNode())
            System.out.println(String.valueOf(getAttribute(attr)));;

        g.display();

    }
}
