package Graph_Test_Ground;

/**
 * Created by Administer on 21/05/2018.
 */



    import java.io.ByteArrayInputStream;
 import java.io.IOException;

 import org.graphstream.algorithm.APSP;
 import org.graphstream.algorithm.APSP.APSPInfo;
 import org.graphstream.algorithm.APSP.Progress;
    import org.graphstream.algorithm.Centroid;
    import org.graphstream.graph.Graph;
    import org.graphstream.graph.Node;
    import org.graphstream.graph.implementations.DefaultGraph;
 import org.graphstream.stream.file.FileSourceDGS;

    /**
     *
     *     B-(1)-C
     *    /       \
     *  (1)       (10)
     *  /           \
     * A             F
     *  \           /
     *  (1)       (1)
     *    \       /
     *     D-(1)-E
     */
    public class Floyd_Warshall {

        static String my_graph =
                "DGS004\n"
                        + "my 0 0\n"
                        + "an A \n"
                        + "an B \n"
                        + "an C \n"
                        + "an D \n"
                        + "an E \n"
                        + "an F \n"
                        + "ae AB A B weight:1 \n"
                        + "ae AD A D weight:1 \n"
                        + "ae BC B C weight:1 \n"
                        + "ae CF C F weight:10 \n"
                        + "ae DE D E weight:1 \n"
                        + "ae EF E F weight:1 \n"
                ;

        public static void main(String[] args) throws IOException {
            Graph graph = new DefaultGraph("APSP Test");


//            ByteArrayInputStream bs = new ByteArrayInputStream(my_graph.getBytes());
//
//            FileSourceDGS source = new FileSourceDGS();
//            source.addSink(graph);
//            source.readAll(bs);

            graph.setAutoCreate(true);
            graph.setStrict(false);
            graph.addEdge("AB","A", "B", true);
            graph.addEdge("BC","B", "C", true);
            graph.addEdge("CD","C", "D", true);
            graph.addEdge("DA","D", "A", true);


            APSP apsp = new APSP();
            apsp.init(graph); // registering apsp as a sink for the graph
            apsp.setDirected(false); // undirected graph
            //apsp.setWeightAttributeName("weight"); // ensure that the attribute name used is "weight"

            apsp.compute(); // the method that actually computes shortest paths
            Centroid centroid = new Centroid();
        centroid.init(graph);
        //centroid.setCentroidAttribute("weight");
        centroid.compute();

        for (Node n : graph.getEachNode()) {
            Boolean in = n.getAttribute("centroid");
            System.out.printf("%s is%s in the centroid.\n", n.getId(), in ? ""
                    : " not");
        }

        }
    }

