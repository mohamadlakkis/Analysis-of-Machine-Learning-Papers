import spacy
import networkx as nx
from pyvis.network import Network
import community # # louvain for community detection
import spacy.lang  

###############################################################################
# 1. LOAD TEXT AND SPLIT INTO PARAGRAPHS
###############################################################################

def load_document(file_path:str) -> list[str]:
    """
        Loads a text document and splits it into paragraphs.

        This function reads the content of a text file specified by the given file path.
        It assumes that paragraphs in the text are separated by blank lines or newline characters.(\n\n) [This is a temporary solution, and you may need to modify this function based on the text file's formatting.]
        The function returns a list of strings, where each string represents a paragraph.

        Parameters:
        file_path (str): The path to the text file to be loaded.

        Returns:
        list: A list of strings, each representing a paragraph from the text file.
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split on double newlines (like our assumption)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs


###############################################################################
# 2. TEXT NORMALIZATION AND STOPWORD REMOVAL (spaCy)
###############################################################################

def preprocess_text(paragraphs: list[str], nlp_model: spacy.lang = spacy.load('en_core_web_sm')) -> list[list[str]]:
    """
        Preprocess a list of paragraphs using spaCy.
        This function applies the following preprocessing steps to each paragraph:
          - Tokenization (word or sentence, depending on the model)[ Default is 'en_core_web_sm' i.e. word tokenization]
          - Lemmatization
          - Punctuation removal
          - Stopword removal
        Args:
            paragraphs (list of str): A list of paragraphs to be processed.
            nlp_model (spacy.lang): A spaCy language model for processing the text. Default is 'en_core_web_sm'.
        Returns:
           
            cleaned_paragraphs (list of list of str): A list of paragraphs, where each paragraph is a list of lemmas (strings).
    """
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        doc = nlp_model(paragraph)
        tokens = []
        for token in doc:
            # Filter out punctuation and stopwords
            if not token.is_punct and not token.is_space and not token.is_stop:
                tokens.append(token.lemma_.lower())
        # No need to add empty lists (no tokens)
        if tokens:
            cleaned_paragraphs.append(tokens)
    
    return cleaned_paragraphs



###############################################################################
# 3. BUILD THE GRAPH (NETWORKX) (Text-to-Network Conversion)
###############################################################################

def add_weighted_edge(G:nx.Graph, source:nx.Graph.nodes, target:nx.Graph.nodes, weight:int)-> None:
    """
    Helper function to add or update a weighted edge in a DiGraph (in-place).
    If the edge already exists, we add the new weight to the existing one. (Like in the paper)
    """
    if G.has_edge(source, target):
        G[source][target]['weight'] += weight
    else:
        G.add_edge(source, target, weight=weight)
        

def build_graph(cleaned_paragraphs: list[list[str]]) -> nx.DiGraph:
    """
        Build a directed, weighted graph based on co-occurrence rules.
        This function constructs a NetworkX DiGraph from a list of cleaned paragraphs.
        The graph is built with the following co-occurrence rules:
        Parameters:
            cleaned_paragraphs (list of list of str): A list where each element is a list of tokens (words) representing a cleaned paragraph. (The one returned by preprocess_text)
        Returns:
            networkx.DiGraph: A directed, weighted graph where nodes represent words and edges represent co-occurrence  relationships with specified weights. (with attributes: frequency)
    """
    
    
    G = nx.DiGraph()
    
    for paragraph in cleaned_paragraphs:
        length = len(paragraph)

        # A. Create nodes first (with a frequency attribute), I will augment this later with community detection, ... 
        for token in paragraph:
            if not G.has_node(token):
                G.add_node(token, frequency=0)
            G.nodes[token]['frequency'] += 1 

        # B. Add edges, based  on the co-occurrence rules from paper
        for i in range(length):
            current_word = paragraph[i]
            # (distance = 1 => weight=3)
            if i+1 < length:
                next_word = paragraph[i+1]
                add_weighted_edge(G, source = current_word, target =  next_word, weight = 3)
            
            # (distance = 2 => weight=2, distance = 3 => weight=1)
            if i+2 < length: # prevent edge cases
                word_2_ahead = paragraph[i+2]
                add_weighted_edge(G, current_word, word_2_ahead, 2)
            if i+3 < length: # prevent edge cases 
                word_3_ahead = paragraph[i+3]
                add_weighted_edge(G, current_word, word_3_ahead, 1)
    
    return G


###############################################################################
# 4. CENTRALITY (BETWEENNESS) AND COMMUNITY DETECTION
###############################################################################

def analyze_graph(G: nx.Graph) -> nx.Graph:
    '''
        Analyzes the given graph by computing betweenness centrality and performing community detection using the Louvain method. (from paper)
        Parameters:
            G (networkx.Graph): A NetworkX graph object, which can be directed or undirected.
        Returns:
            networkx.Graph: The input graph with augmented node attributes for betweenness centrality and community detection.
        The function performs the following steps:
            0. A preprocessing step is necessary to augment the weights with distances, since IN OUR EXAMPLE a path should prefer shorter distances (BUT LARGER WEIGHTS), since the weights are based on co-occurrence frequency. And so distance = 1/weight.
            1. Computes the betweenness centrality for each node in the graph and stores it as a node attribute 'betweenness'.
            2. Converts a copy of the graph to an undirected version for community detection.
            3. Applies the Louvain method for community detection and stores the resulting community ID as a node attribute 'community'.
    '''
    # 0. Convert weights to distances (since we want to prefer shorter distances)
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 0)  # default to 0 if somehow no weight
        if w > 0:
            data['distance'] = 1 / w
        else:
            data['distance'] = float('inf')

    # A. Betweenness Centrality
    betweenness_dict = nx.betweenness_centrality(G, normalized=True, weight='distance') # notice how here we use distance instead of weight (since we want to prefer shorter distances)(but larger weights)
    # Store betweenness as a node attribute
    for node, bc_value in betweenness_dict.items():
        G.nodes[node]['betweenness'] = bc_value
    
    # B. Community Detection (Louvain) - requires an undirected graph since Louvain is for undirected graphs
    undirected_G = G.to_undirected() 
    partition = community.best_partition(undirected_G, weight='weight')
    # Store community ID as a node attribute
    for node, comm_id in partition.items():
        G.nodes[node]['community'] = comm_id
    
    return G


###############################################################################
# 5. INTERACTIVE VISUALIZATION (PYVIS) later I will move to gephi/Force-Atlas for more advanced visualization
###############################################################################

def visualize_graph_pyvis(G:nx.graph, output_html : str ='graph_visualization_pyvis.html') -> None:
    """
        Create an interactive force-directed graph visualization using the PyVis library.
        This function generates an HTML file containing an interactive visualization of the given NetworkX graph.
        The visualization mimics a Force-Atlas layout by adjusting PyVis physics parameters.
        Parameters:
        -----------
            G : nx.Graph
                A NetworkX graph object to be visualized.
            output_html : str, optional
                The filename for the output HTML file (default is 'graph_visualization.html').
        Returns:
        --------
            None
        Notes:
        ------
            - Nodes are added to the visualization with sizes scaled by their 'frequency' attribute and colored by their 'community' attribute.
            - Edges are added with thickness scaled by their 'weight' attribute.
            - Tooltips for nodes display their 'betweenness' and 'frequency' attributes.
            - The visualization uses a Force-Atlas 2-based layout with customizable physics parameters for gravity, central gravity, spring length, spring strength, and damping.

        Mechaniques: 
            - I am coloring the nodes by community for now, later I will add more advanced visualization techniques
            - I am scaling the node size by frequency for now
            - I am scaling the edge thickness by weight for now


    """
    
    net = Network(notebook=True, directed=True, width='1800px', height='1000px') 
    
    # For a “Force-Atlas”-like effect, feel free to adjust these parameters
    net.force_atlas_2based(
        gravity=-50,           # negative gravity means "repulsive"
        central_gravity=0.005, # how strongly nodes are pulled to the center
        spring_length=100,     # length of edges
        spring_strength=0.01,  # how much edges act like springs
        damping=0.9            # speed of the simulation
    )
    
    # Add nodes and edges from NetworkX graph to PyVis network 
    for node, data in G.nodes(data=True):
        comm_id = data.get('community', 90) # default to 90 if no community
        bc_value = data.get('betweenness', 0.0) # default to 0.0 if no betweenness
        freq = data.get('frequency', 1) # default to 1 if no frequency (shouldn't happen)
        
        # scale node size by frequency
        th =  10
        size = th + freq*4
        
        # color by community
        color = community_color(comm_id)
        
        title = f"Betweenness: {bc_value:.4f}, Frequency: {freq}" # When a user hovers over a node or edge, the tooltip text will be displayed
        
        net.add_node(
            n_id=node,
            label=node,
            title=title,
            size=size,
            color=color,
            font={'size': bc_value*1000+10 }
        )   
    
    for u, v, w_data in G.edges(data=True):
        weight = w_data['weight']
        # if weight <= 3:
        #     continue
        # scale edge thickness by weight
        net.add_edge(u, v, value=weight, title=f"Weight: {weight}")
    
    net.show(output_html)
    print(f"Interactive graph saved to {output_html}")


def community_color(community_id: int) -> str:
    """
        Maps a given community ID to a corresponding color.
            This function uses a predefined list of colors and maps the given 
            community ID to one of these colors. The mapping is done using the 
            modulo operation to ensure that the community ID wraps around if it 
            exceeds the number of available colors.
        Args:
            community_id (int): The ID of the community to be mapped to a color.
        Returns:
            str: The hexadecimal color code corresponding to the given community ID.
        Notes: 
            Later I will use more advanced visualization techniques
    """
    colors = [
        "#FF595E"   # red
        , "#FFCA3A" # yellow
        , "#8AC926" # green
        , "#1982C4" # blue
        , "#6A4C93" # purple
        , "#FF9B85" # pink
        , "#00A676" # teal
        , "#FFA400" # orange
    ]
    return colors[community_id % len(colors)]




###############################################################################
# MAIN EXAMPLE USAGE
###############################################################################

if __name__ == "__main__":
    # 1. LOAD DOCUMENT
    # Provide your text file path here:
    text_file_path = "texts/texts_ghandi_entries.txt"  
    paragraphs = load_document(text_file_path)
    
    # Initialize spaCy
    nlp = spacy.load("en_core_web_md")
    
    # 2. PREPROCESS (lemmatize, remove stop words, punctuation)
    cleaned_paragraphs = preprocess_text(paragraphs, nlp)
    
    # 3. BUILD THE GRAPH
    G = build_graph(cleaned_paragraphs)
    
    # 4. CENTRALITY & COMMUNITY DETECTION
    G = analyze_graph(G)
    
    # 5. VISUALIZE INTERACTIVELY (Force-Atlas style)
    visualize_graph_pyvis(G)