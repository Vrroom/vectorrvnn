import networkx as nx
import json

class GraphReadWrite () :
    """
    Convenience class to handle graph
    read/write operations so that
    I don't have to keep searching 
    for networkx documentation.

    Two graph types are supported right now:
        1) 'cytoscape'
        2) 'tree'
    """

    def __init__ (self, graphType) :
        """
        Parameters
        ----------
        graphType : str
            Indicate which graph type
            to read/write.
        """
        self.graphType = graphType

    def read (self, inFile) :
        """
        Parameters
        ----------
        inFile : str
            Path to input file to write to.
        """
        with open(inFile, 'r') as fd :
            dct = json.loads(fd.read())
        if self.graphType == 'cytoscape' :
            return nx.readwrite.json_graph.cytoscape_graph(dct)
        elif self.graphType == 'tree' :
            return nx.readwrite.json_graph.tree_graph(dct)
        
        raise ValueError ('Unsupported Graph Type')

    def write (self, G, outFile) :
        """
        Parameters
        ----------
        G : nx.Graph / (nx.DiGraph, int)
            If graphType is cytoscape then
            it is a nx.Graph, else, it is the 
            tree along with the root index.
        outFile : str
            Path to the output file.
        """
        dct = None
        if self.graphType == 'cytoscape' :
            dct = nx.readwrite.json_graph.cytoscape_data(G)
        elif self.graphType == 'tree' :
            T, r = G
            dct = nx.readwrite.json_graph.tree_data(T, r)

        if dct is None :
            raise ValueError ('Unsupported Graph Type')

        with open(outFile, 'w+', encoding='utf-8') as fd :
            json.dump(dct, fd, ensure_ascii=False, indent=4)


