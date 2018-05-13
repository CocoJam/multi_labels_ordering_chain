package Graph_Test_Ground;

/**
 * Created by Administer on 27/04/2018.
 */

import org.graphstream.graph.*;
import org.graphstream.graph.implementations.*;



public class graph_play
{
    public static void main(String[] args) {


        Graph graph = new SingleGraph("Tutorial 1");

        graph.setStrict(false);
        graph.setAutoCreate( true );
        graph.addEdge("AB", "A", "B");
        graph.addEdge("BC", "B", "C");
        graph.addEdge("CA", "C", "A");

        Node A = graph.getNode("A");
        Edge E =graph.getEdge("AB");

        /*
        what is the unique identifier of the node or edge (Node.getId(), Edge.getId()),
        what is the degree of the node (number of connected edges, Node.getDegree()),
        what are the edges connected to a node (Node.hasEdgeToward(String id) and Node.getEdgeToward(String id)),
        is an edge directed or not (Edge.isDirected()),
        what is the node at the other end of the edge (Edge.getOpposite(Node n)),
        etc.
         */


        for(Node n:graph) {
            System.out.println(n.getId());
        }


        int n = graph.getNodeCount();
        byte adjacencyMatrix[][] = new byte[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                adjacencyMatrix[i][j] = (byte) (graph.getNode(i).hasEdgeBetween(j) ? 1 : 0);

        System.out.println(adjacencyMatrix[1][1]);


        graph.display();



    }







}
