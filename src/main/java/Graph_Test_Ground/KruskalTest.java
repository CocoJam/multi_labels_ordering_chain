package Graph_Test_Ground;

/**
 * Created by Administer on 19/05/2018.
 */
import org.graphstream.graph.Graph;
import org.graphstream.graph.implementations.DefaultGraph;

import org.graphstream.algorithm.Kruskal;
import org.graphstream.algorithm.generator.DorogovtsevMendesGenerator;

public class KruskalTest {
    public static void main(String ... args) {
        DorogovtsevMendesGenerator gen = new DorogovtsevMendesGenerator();
        Graph graph = new DefaultGraph("Kruskal Test");

        String css = "edge .notintree {size:1px;fill-color:gray;} " +
                "edge .intree {size:3px;fill-color:black;}";

        graph.addAttribute("ui.stylesheet", css);
        graph.display();

        gen.addEdgeAttribute("weight");
        gen.setEdgeAttributesRange(1, 100);
        gen.addSink(graph);
        gen.begin();
        for (int i = 0; i < 100 && gen.nextEvents(); i++)
            ;
        gen.end();

        Kruskal kruskal = new Kruskal("ui.class", "intree", "notintree");

        kruskal.init(graph);
        kruskal.compute();
    }
}
