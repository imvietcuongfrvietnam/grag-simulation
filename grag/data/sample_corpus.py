"""
Rich sample corpus for Graph RAG simulation.

Contains 25 documents about a fictional tech company ecosystem with natural
entity relationships woven consistently across documents.
"""

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "NeuralTech Corporation Overview",
        "content": (
            "NeuralTech Corporation is a leading artificial intelligence company "
            "headquartered in San Francisco, California. Founded in 2015 by Dr. Elena "
            "Vasquez and Dr. Marcus Chen, NeuralTech has grown from a small research lab "
            "into a global enterprise employing over 3,000 engineers and scientists. The "
            "company flagship product, NeuroGraph, is an enterprise knowledge graph "
            "platform that enables organizations to build and query large-scale knowledge "
            "graphs using transformer architectures and graph neural networks. "
            "Dr. Elena Vasquez serves as Chief Executive Officer and has guided the "
            "company through three major funding rounds, raising a total of 420 million "
            "dollars. Dr. Marcus Chen, who serves as Chief Technology Officer, previously "
            "worked at DeepMind before co-founding NeuralTech. The company research division, "
            "NeuralTech Research Lab, is led by Dr. Priya Nair, a renowned expert in "
            "graph representation learning. "
            "NeuralTech core technology stack relies heavily on PyTorch, CUDA-accelerated "
            "GPU clusters, and its proprietary GraphStream processing engine. The company "
            "has published over 40 peer-reviewed papers and holds 28 patents related to "
            "knowledge graph construction and retrieval. In 2022, NeuralTech acquired "
            "GraphIndex Labs, a startup specializing in distributed graph indexing, "
            "significantly expanding its infrastructure capabilities. "
            "The company operates data centers in San Francisco, London, and Singapore, "
            "serving enterprise clients across finance, healthcare, and legal sectors. "
            "NeuralTech annual revenue exceeded 180 million dollars in 2023, with a "
            "year-over-year growth rate of 47 percent, making it one of the fastest-growing "
            "AI companies in the knowledge graph space."
        ),
    },
    {
        "id": "doc_002",
        "title": "QuantumAI Systems: Company Profile",
        "content": (
            "QuantumAI Systems is an enterprise AI platform company based in Austin, Texas, "
            "founded in 2017 by Dr. James Okafor and Sarah Mitchell. The company specializes "
            "in large language model fine-tuning, retrieval-augmented generation systems, "
            "and enterprise search infrastructure. QuantumAI flagship platform, "
            "QuantumSearch, competes directly with NeuralTech NeuroGraph in the enterprise "
            "knowledge management market. "
            "Dr. James Okafor, a former researcher at OpenAI, serves as CEO of QuantumAI. "
            "Sarah Mitchell, the company President and co-founder, previously held the "
            "position of VP of Engineering at DataVerse Inc before leaving to launch "
            "QuantumAI. The company has raised 290 million dollars across its Series A, "
            "B, and C funding rounds, with its most recent valuation reaching 2.1 billion "
            "dollars. "
            "QuantumAI engineering team pioneered a novel retrieval method called Sparse "
            "Dense Fusion, which combines BM25 keyword retrieval with dense vector search "
            "using transformer-based embeddings. This technique has been widely cited in "
            "academic literature and forms the backbone of QuantumSearch relevance engine. "
            "In 2023, QuantumAI partnered with DataVerse Inc to integrate QuantumSearch "
            "into DataVerse enterprise data warehouse platform, creating a jointly marketed "
            "solution called DataVerse Intelligence. The partnership was announced at the "
            "TechSummit 2023 conference in Las Vegas. QuantumAI currently employs 850 people "
            "and operates offices in Austin, New York, and London."
        ),
    },
    {
        "id": "doc_003",
        "title": "DataVerse Inc.: Data Infrastructure Giant",
        "content": (
            "DataVerse Inc is one of the world largest enterprise data infrastructure "
            "companies, headquartered in Seattle, Washington. Founded in 2009 by Robert "
            "Tanaka and Dr. Anika Sharma, DataVerse provides cloud-native data warehousing, "
            "data lake management, and real-time streaming analytics to over 8,000 enterprise "
            "clients worldwide. The company went public on the NASDAQ exchange in 2019 and "
            "currently holds a market capitalization of approximately 34 billion dollars. "
            "Robert Tanaka serves as Chairman of the Board while Dr. Anika Sharma acts as "
            "CEO, having assumed the role in 2021. DataVerse core product suite includes "
            "DataVerse Cloud, DataVerse Stream, and DataVerse Catalog, which collectively "
            "handle over 2 exabytes of data daily for clients in sectors ranging from "
            "financial services to logistics. "
            "DataVerse partnered with QuantumAI Systems in 2023 to launch DataVerse "
            "Intelligence, an AI-powered search and analytics layer built on top of "
            "QuantumAI QuantumSearch platform. This partnership represented DataVerse "
            "strategic move into AI-augmented data discovery. DataVerse also collaborated "
            "with NeuralTech Corporation to evaluate NeuroGraph for internal knowledge "
            "management in 2022. DataVerse employs over 12,000 people globally and reported "
            "revenue of 4.2 billion dollars in fiscal year 2023."
        ),
    },
    {
        "id": "doc_004",
        "title": "TechFlow Inc. and the Road to Acquisition",
        "content": (
            "TechFlow Inc was a pioneering natural language processing company based in "
            "Boston, Massachusetts, founded in 2013 by Dr. Lena Park and Professor Alan "
            "Winters. The company built advanced NLP pipelines for document understanding, "
            "named entity recognition, and relation extraction. TechFlow proprietary "
            "toolkit, FlowNLP, gained wide adoption in the legal tech and healthcare "
            "informatics communities. "
            "The company early success attracted the attention of CogniSystems, a larger "
            "enterprise AI firm that had been building its own knowledge graph capabilities. "
            "After months of negotiation, CogniSystems announced in September 2022 the "
            "acquisition of TechFlow Inc for 650 million dollars, the largest acquisition "
            "in the NLP tools market that year. Dr. Lena Park joined CogniSystems as "
            "Chief Science Officer following the merger, while Professor Alan Winters "
            "returned to his academic position at MIT. "
            "The acquisition was driven by CogniSystems desire to integrate FlowNLP "
            "entity extraction capabilities into its CogniGraph product line. Several key "
            "engineers from TechFlow, including Dr. Wei Zhang and Maya Rodriguez, relocated "
            "to CogniSystems headquarters in Chicago following the merger. The integration "
            "was completed in early 2023, with FlowNLP technology now deeply embedded "
            "in CogniGraph 3.0. "
            "Industry analysts noted that TechFlow acquisition by CogniSystems was partly "
            "triggered by increasing competitive pressure from NeuralTech Corporation, whose "
            "NeuroGraph platform had begun capturing TechFlow key accounts in the healthcare "
            "vertical. The deal was seen as a defensive move by CogniSystems to bolster its "
            "NLP capabilities against NeuralTech."
        ),
    },
    {
        "id": "doc_005",
        "title": "CogniSystems: Enterprise Knowledge Graph Pioneer",
        "content": (
            "CogniSystems is an enterprise AI and knowledge management company headquartered "
            "in Chicago, Illinois. Founded in 2010 by Dr. Howard Bell and Dr. Fatima Al-Rashid, "
            "CogniSystems was among the first companies to commercialize knowledge graph "
            "technology for enterprise use cases. The company flagship product, CogniGraph, "
            "serves over 500 Fortune 1000 clients and has been deployed in industries including "
            "pharmaceuticals, finance, and manufacturing. "
            "Dr. Howard Bell serves as CEO and has steered CogniSystems through significant "
            "market evolution, including the rise of large language models and the subsequent "
            "boom in graph-augmented AI systems. Dr. Fatima Al-Rashid, now Chief Research "
            "Officer, leads the company AI research division and has published extensively "
            "on ontology design and knowledge representation. "
            "CogniSystems acquired TechFlow Inc in September 2022 for 650 million dollars, "
            "integrating FlowNLP capabilities into CogniGraph 3.0. This acquisition "
            "significantly strengthened CogniSystems position in the competitive landscape "
            "against NeuralTech Corporation and QuantumAI Systems. Following the acquisition, "
            "Dr. Lena Park from TechFlow joined CogniSystems as Chief Science Officer. "
            "CogniSystems uses transformer architectures extensively in its entity resolution "
            "and link prediction pipelines. The company participated in the KnowledgeGraph "
            "Summit 2023 alongside NeuralTech and QuantumAI, where all three firms demonstrated "
            "graph-based retrieval-augmented generation systems."
        ),
    },
    {
        "id": "doc_006",
        "title": "Dr. Elena Vasquez: Visionary in AI",
        "content": (
            "Dr. Elena Vasquez is one of the most prominent figures in the artificial "
            "intelligence industry, best known as co-founder and CEO of NeuralTech Corporation. "
            "Born in Mexico City and educated at Stanford University, where she completed her "
            "PhD in Computer Science under Professor Michael Huang, Dr. Vasquez doctoral "
            "research focused on graph-structured neural networks and their application to "
            "knowledge representation. "
            "Before founding NeuralTech in 2015 alongside Dr. Marcus Chen, Dr. Vasquez held "
            "a research scientist position at Google Brain, where she contributed to early "
            "work on graph neural networks and attention mechanisms. Her landmark paper "
            "Hierarchical Graph Attention for Knowledge Base Completion published in 2014 "
            "remains among the most cited works in the knowledge graph research community. "
            "Dr. Vasquez serves on the board of directors of the AI Safety Institute and is "
            "a founding member of the Responsible AI Coalition. She has been recognized in "
            "Forbes list of America Most Powerful Women in Technology for three consecutive "
            "years. In 2023, she delivered a keynote address at TechSummit 2023 in Las Vegas, "
            "presenting NeuralTech vision for the next generation of enterprise knowledge graphs. "
            "Outside of NeuralTech, Dr. Vasquez is an angel investor who has backed several "
            "early-stage AI startups, including KGraphCo and SynthData Labs. She is known "
            "for her advocacy for open-source AI tooling and contributed to the early "
            "development of the PyTorch Geometric library."
        ),
    },
    {
        "id": "doc_007",
        "title": "Dr. Marcus Chen: Engineering the Knowledge Graph Future",
        "content": (
            "Dr. Marcus Chen is the co-founder and Chief Technology Officer of NeuralTech "
            "Corporation, a role he has held since the company founding in 2015. Prior to "
            "NeuralTech, Dr. Chen spent five years as a research scientist at DeepMind in "
            "London, working on graph-based reasoning systems and reinforcement learning "
            "applied to combinatorial optimization. His work at DeepMind included co-authoring "
            "the influential paper Graph Networks as Learnable Physics Simulators. "
            "At NeuralTech, Dr. Chen led the architecture of GraphStream, the company "
            "proprietary distributed graph processing engine that powers NeuroGraph. He "
            "assembled NeuralTech core engineering team, recruiting heavily from top "
            "universities and research institutions. Under his technical leadership, "
            "NeuralTech has developed a suite of graph machine learning models including "
            "GraphBERT, a BERT-based model adapted for entity disambiguation in knowledge graphs. "
            "Dr. Chen collaborated with Professor Yuki Tanaka from Stanford University on "
            "the development of the GraphRAG benchmark suite, which has become a standard "
            "evaluation framework for graph-based retrieval augmented generation systems. "
            "This collaboration produced three joint publications and an open-source "
            "evaluation toolkit hosted on GitHub. "
            "Dr. Chen is frequently invited to speak at academic conferences including "
            "NeurIPS, ICLR, and the KnowledgeGraph Summit. He serves as an adjunct professor "
            "at UC Berkeley, where he teaches a graduate seminar on large-scale graph systems."
        ),
    },
    {
        "id": "doc_008",
        "title": "GraphIndex Labs: The Acquisition Story",
        "content": (
            "GraphIndex Labs was a distributed systems startup specializing in high-performance "
            "graph indexing and traversal algorithms. Founded in 2018 in San Francisco by "
            "Dr. Nadia Petrov and Carlos Mendez, the company developed a novel disk-based "
            "graph index called SpiderIndex that could process trillion-edge graphs with "
            "millisecond query latency. SpiderIndex attracted significant attention from "
            "major enterprise technology companies. "
            "Dr. Nadia Petrov, who holds a PhD in distributed systems from Carnegie Mellon "
            "University, served as CEO of GraphIndex Labs. Carlos Mendez, a systems engineer "
            "formerly at Databricks, was the CTO. The company raised 45 million dollars in "
            "two funding rounds before NeuralTech Corporation acquired GraphIndex Labs in "
            "2022 for approximately 180 million dollars. "
            "Following the acquisition, SpiderIndex was integrated into NeuralTech "
            "GraphStream engine, dramatically improving NeuroGraph ability to handle "
            "large-scale enterprise knowledge graphs. Dr. Nadia Petrov joined NeuralTech "
            "as VP of Infrastructure, while Carlos Mendez became a Senior Principal Engineer "
            "on the NeuroGraph platform team. "
            "The acquisition of GraphIndex Labs by NeuralTech was a strategic move to prevent "
            "competing firms, particularly CogniSystems, from acquiring the technology. "
            "CogniSystems had reportedly been in late-stage acquisition discussions with "
            "GraphIndex Labs before NeuralTech made a superior offer. This acquisition "
            "intensified the competitive rivalry between NeuralTech and CogniSystems."
        ),
    },
    {
        "id": "doc_009",
        "title": "TechSummit 2023: AI and Knowledge Graph Trends",
        "content": (
            "TechSummit 2023 was a major technology conference held in Las Vegas, Nevada in "
            "October 2023, drawing over 15,000 attendees from across the global technology "
            "industry. The event was organized by the Global Technology Forum and featured "
            "keynote addresses from leaders across AI, cloud computing, and enterprise software. "
            "The conference was notable for a major announcement from QuantumAI Systems and "
            "DataVerse Inc, who jointly unveiled DataVerse Intelligence, their co-developed "
            "AI-powered enterprise search platform. CEO Dr. James Okafor of QuantumAI and "
            "CEO Dr. Anika Sharma of DataVerse jointly presented the product, emphasizing "
            "its integration with DataVerse existing data infrastructure. "
            "Dr. Elena Vasquez of NeuralTech Corporation delivered a widely praised keynote "
            "titled From Documents to Knowledge: The Future of Enterprise Intelligence, "
            "in which she demonstrated NeuroGraph new multi-modal knowledge graph capabilities. "
            "She argued that transformer architectures alone are insufficient for enterprise "
            "reasoning tasks and that explicit graph structure is essential. "
            "The KnowledgeGraph Summit, a co-located event at TechSummit 2023, brought "
            "together researchers and practitioners to discuss advances in graph-based AI. "
            "Notable speakers included Dr. Priya Nair from NeuralTech Research Lab, "
            "Dr. Fatima Al-Rashid from CogniSystems, and Professor Yuki Tanaka from "
            "Stanford University. The summit featured live demonstrations of Microsoft "
            "GraphRAG, RAPTOR, and HippoRAG systems side by side."
        ),
    },
    {
        "id": "doc_010",
        "title": "KnowledgeGraph Summit 2023: Research Highlights",
        "content": (
            "The KnowledgeGraph Summit 2023, held as a co-located event with TechSummit 2023 "
            "in Las Vegas, was a three-day technical conference dedicated to advances in "
            "knowledge graph construction, reasoning, and retrieval. The summit attracted "
            "over 800 researchers and engineers and featured 60 paper presentations, six "
            "workshops, and four invited keynote talks. "
            "Professor Yuki Tanaka from Stanford University delivered the opening keynote "
            "on Benchmarking Graph Retrieval Augmented Generation, presenting results "
            "from the GraphRAG benchmark suite he co-developed with Dr. Marcus Chen of "
            "NeuralTech Corporation. The benchmark compared Microsoft GraphRAG, RAPTOR, "
            "HippoRAG, and several other systems on multi-hop reasoning tasks. "
            "Dr. Priya Nair from NeuralTech Research Lab presented work on scalable "
            "community detection algorithms for large knowledge graphs, comparing the "
            "Louvain, Leiden, and Label Propagation methods on real-world enterprise "
            "knowledge graphs. Her results showed Leiden achieving 12 percent higher "
            "modularity scores than Louvain on sparse graphs. "
            "A workshop on Hybrid Retrieval Systems featured contributions from researchers "
            "at NeuralTech, CogniSystems, and the University of Cambridge. Dr. Fatima "
            "Al-Rashid from CogniSystems presented CogniGraph approach to combining "
            "dense retrieval with explicit knowledge graph traversal, arguing that hybrid "
            "methods consistently outperform pure vector retrieval on complex reasoning queries."
        ),
    },
    {
        "id": "doc_011",
        "title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "content": (
            "RAPTOR is a novel document indexing and retrieval system developed by researchers "
            "at Stanford University, with key contributions from Professor Yuki Tanaka and "
            "Dr. Sofia Romero. The system builds a hierarchical tree structure over document "
            "collections by recursively clustering and summarizing text chunks, enabling "
            "retrieval at multiple levels of abstraction. "
            "The core innovation of RAPTOR lies in its tree-based indexing approach. Documents "
            "are first chunked into passages, then encoded using transformer-based embeddings. "
            "Gaussian mixture models cluster the passages at each level, and a summarization "
            "model generates abstract summaries of each cluster. This process repeats "
            "recursively until a single root node summarizes the entire corpus. "
            "At query time, RAPTOR supports two retrieval strategies: tree traversal, which "
            "navigates from root to leaf using semantic similarity at each level, and collapsed "
            "retrieval, which treats all nodes in the tree as a flat pool and selects the "
            "most relevant across all levels simultaneously. "
            "RAPTOR was benchmarked against Microsoft GraphRAG, HippoRAG, and standard RAG "
            "baselines in the GraphRAG benchmark suite developed by Professor Yuki Tanaka "
            "and Dr. Marcus Chen from NeuralTech Corporation. RAPTOR demonstrated particularly "
            "strong performance on thematic and abstractive queries. "
            "The RAPTOR codebase is open-source and has been integrated into several "
            "commercial products, including an experimental version within NeuralTech "
            "NeuroGraph platform. Dr. Sofia Romero subsequently joined NeuralTech Research "
            "Lab as a visiting researcher."
        ),
    },
    {
        "id": "doc_012",
        "title": "HippoRAG: Neurologically Inspired RAG Framework",
        "content": (
            "HippoRAG is a retrieval-augmented generation framework inspired by the human "
            "hippocampal memory system, developed by Dr. Zhen Liu and colleagues at the "
            "University of Cambridge. The system uses a knowledge graph backbone to simulate "
            "the pattern separation and pattern completion mechanisms observed in hippocampal "
            "episodic memory, allowing for highly associative retrieval across documents. "
            "The HippoRAG architecture consists of three main components: the Parahippocampal "
            "Cortex module for initial passage encoding, the Dentate Gyrus module for pattern "
            "separation via sparse entity extraction, and the CA3 module for pattern completion "
            "via graph traversal. These components work together to enable multi-hop associative "
            "retrieval that generalizes beyond simple keyword or semantic matching. "
            "Dr. Zhen Liu collaborated with Dr. Fatima Al-Rashid from CogniSystems during "
            "the development of HippoRAG knowledge graph construction pipeline. This "
            "collaboration led to a joint paper presented at the KnowledgeGraph Summit 2023 "
            "comparing HippoRAG with CogniGraph hybrid retrieval approach. "
            "HippoRAG was evaluated against RAPTOR, Microsoft GraphRAG, and standard dense "
            "retrieval baselines in the GraphRAG benchmark developed by Professor Yuki Tanaka "
            "and Dr. Marcus Chen. The system showed particularly strong performance on "
            "multi-hop reasoning tasks where evidence must be connected across multiple "
            "separate documents."
        ),
    },
    {
        "id": "doc_013",
        "title": "Microsoft GraphRAG: Global and Local Search Architecture",
        "content": (
            "Microsoft GraphRAG is an open-source framework developed by Microsoft Research "
            "that applies graph machine learning to retrieval-augmented generation. The system "
            "was designed to address the limitations of standard RAG approaches on global, "
            "thematic questions that require synthesizing information across an entire document "
            "corpus rather than retrieving specific passages. "
            "Microsoft GraphRAG indexing pipeline extracts entities and relationships from "
            "documents using large language models, builds a knowledge graph, detects "
            "communities using the Leiden algorithm, and generates hierarchical community "
            "summaries at multiple granularity levels. These summaries form the backbone "
            "of the global search capability. "
            "The system supports two distinct search modes: global search performs a "
            "map-reduce operation over community summaries to answer broad questions about "
            "the corpus, while local search expands from specific seed entities through the "
            "knowledge graph to answer focused, entity-centric queries. "
            "Microsoft GraphRAG was benchmarked at the KnowledgeGraph Summit 2023 alongside "
            "RAPTOR, HippoRAG, and other systems. Dr. Priya Nair from NeuralTech Research "
            "Lab ran experiments comparing community detection methods within Microsoft "
            "GraphRAG and found that Leiden-based communities produced higher-quality summaries "
            "than Louvain communities due to better internal connectivity guarantees. "
            "NeuralTech NeuroGraph platform drew heavily on Microsoft GraphRAG design "
            "when implementing its own global search feature."
        ),
    },
    {
        "id": "doc_014",
        "title": "Dr. Priya Nair: Graph Machine Learning Researcher",
        "content": (
            "Dr. Priya Nair is the Director of NeuralTech Research Lab, the academic-style "
            "research division of NeuralTech Corporation. A specialist in graph representation "
            "learning, Dr. Nair completed her PhD at the Indian Institute of Technology "
            "Bombay before joining NeuralTech in 2018 at the invitation of Dr. Marcus Chen. "
            "At NeuralTech Research Lab, Dr. Nair leads a team of 35 researchers working "
            "on graph neural networks, community detection, and knowledge graph embeddings. "
            "Her most cited work includes Adaptive Community Detection for Dynamic Knowledge "
            "Graphs, which introduced a novel streaming Louvain variant that handles graph "
            "updates without full recomputation. "
            "Dr. Nair collaborated with Professor Yuki Tanaka from Stanford University on "
            "multiple projects, including contributions to the GraphRAG benchmark suite. "
            "She also worked with Dr. Zhen Liu from the University of Cambridge on a "
            "comparative analysis of HippoRAG and NeuralTech own retrieval system. "
            "At the KnowledgeGraph Summit 2023, Dr. Nair presented findings comparing the "
            "Louvain, Leiden, and Label Propagation community detection algorithms on "
            "enterprise knowledge graphs. Her presentation emphasized that Leiden refinement "
            "phase produces well-connected communities that yield significantly better "
            "retrieval quality in Microsoft GraphRAG-style global search operations. "
            "Dr. Nair is a member of the editorial board of the Journal of Graph Learning "
            "and serves on the program committee of NeurIPS, ICML, and the International "
            "Semantic Web Conference."
        ),
    },
    {
        "id": "doc_015",
        "title": "Professor Yuki Tanaka: Benchmarking Graph RAG Systems",
        "content": (
            "Professor Yuki Tanaka is an associate professor in the Computer Science "
            "Department at Stanford University, specializing in information retrieval, "
            "knowledge graph systems, and evaluation methodology for AI systems. "
            "He is best known for co-developing the GraphRAG benchmark suite in collaboration "
            "with Dr. Marcus Chen from NeuralTech Corporation. "
            "The GraphRAG benchmark, released in early 2023, provides standardized datasets "
            "and evaluation protocols for comparing graph-based retrieval augmented generation "
            "systems. The benchmark includes tasks at varying complexity levels, from simple "
            "entity lookup to complex multi-hop reasoning, and covers diverse domains including "
            "biomedical literature, legal documents, and corporate filings. "
            "Professor Tanaka research group at Stanford, the Knowledge Systems Lab, "
            "developed RAPTOR in collaboration with Dr. Sofia Romero. The group has ongoing "
            "collaborations with NeuralTech Research Lab directed by Dr. Priya Nair, and "
            "has received research funding from both NeuralTech Corporation and DataVerse Inc. "
            "At TechSummit 2023 and the KnowledgeGraph Summit 2023, Professor Tanaka "
            "presented benchmark results comparing Microsoft GraphRAG, RAPTOR, HippoRAG, "
            "KGRAG, MindMapRAG, EdgeRAG, and GRetriever. His analysis showed that no single "
            "system dominates across all query types, and that the choice of algorithm "
            "should be guided by the specific characteristics of the corpus and query distribution."
        ),
    },
    {
        "id": "doc_016",
        "title": "Transformer Architectures in Knowledge Graph Systems",
        "content": (
            "Transformer architectures have become the dominant paradigm for encoding "
            "textual information in modern knowledge graph systems. Originally introduced "
            "by Vaswani et al in 2017, transformers revolutionized natural language "
            "processing and have since been adapted for graph-structured data through "
            "models like GraphBERT, GT-Transformer, and Graphormer. "
            "NeuralTech Corporation has invested heavily in transformer architectures for "
            "its NeuroGraph platform. Dr. Marcus Chen and the NeuralTech engineering team "
            "developed GraphBERT, a transformer variant specifically designed for entity "
            "disambiguation in knowledge graphs. GraphBERT uses graph positional encodings "
            "derived from structural features such as node degree, betweenness centrality, "
            "and community membership to augment standard token embeddings. "
            "QuantumAI Systems Sparse Dense Fusion method also relies on transformer "
            "architectures for the dense retrieval component, using bi-encoder and "
            "cross-encoder transformers for candidate generation and re-ranking respectively. "
            "CogniSystems similarly uses transformer architectures in CogniGraph 3.0, "
            "particularly for the entity resolution pipeline inherited from TechFlow "
            "FlowNLP toolkit. "
            "Microsoft GraphRAG uses large language models, themselves based on transformer "
            "architectures, for both entity extraction during indexing and answer synthesis "
            "during query processing. RAPTOR uses transformer-based sentence encoders to "
            "compute embeddings for its tree construction clustering step."
        ),
    },
    {
        "id": "doc_017",
        "title": "KGRAG: Knowledge Graph Retrieval Augmented Generation",
        "content": (
            "KGRAG is a retrieval-augmented generation framework developed by Dr. Ali Hassan "
            "and his team at the MIT Computer Science and Artificial Intelligence Laboratory. "
            "The system directly grounds language model outputs in a structured "
            "knowledge graph, using explicit triple-based evidence to constrain and verify "
            "generated answers. "
            "Unlike Microsoft GraphRAG or RAPTOR, which build intermediate summarization "
            "structures, KGRAG operates directly on raw knowledge graph triples. The system "
            "uses a graph query planner to decompose natural language questions into sequences "
            "of structured SPARQL-like queries, retrieves relevant subgraphs, and provides "
            "these as structured context to a language model for answer generation. "
            "Dr. Ali Hassan collaborated with Professor Yuki Tanaka from Stanford University "
            "to include KGRAG in the GraphRAG benchmark suite. The benchmark results showed "
            "that KGRAG excels at factual precision queries where exact entity attributes "
            "or relationships are requested, but performs less well on thematic or "
            "abstractive queries that require synthesizing general trends across many entities. "
            "KGRAG was demonstrated at the KnowledgeGraph Summit 2023, where it attracted "
            "interest from CogniSystems researchers. Dr. Fatima Al-Rashid noted that CogniGraph "
            "next version would incorporate elements of KGRAG triple-based evidence grounding."
        ),
    },
    {
        "id": "doc_018",
        "title": "MindMapRAG and EdgeRAG: Novel Graph Traversal Approaches",
        "content": (
            "MindMapRAG and EdgeRAG are two recently proposed graph-based retrieval systems "
            "that take distinct approaches to leveraging graph structure for question answering. "
            "Both systems were presented at the KnowledgeGraph Summit 2023 and evaluated "
            "in the GraphRAG benchmark suite. "
            "MindMapRAG, developed by Dr. Yuna Kim and colleagues at Seoul National University, "
            "builds an explicit mind-map style graph where nodes represent key concepts and "
            "edges represent semantic relationships extracted by an LLM. At query time, the "
            "system performs iterative graph expansion from query-related seed nodes, following "
            "edges weighted by semantic relevance. MindMapRAG uses transformer architectures "
            "for both initial concept extraction and relevance scoring during traversal. "
            "EdgeRAG, developed by Dr. Rafael Torres and his team at ETH Zurich, takes a "
            "fundamentally edge-centric approach. Rather than representing knowledge as node "
            "embeddings, EdgeRAG encodes relationship triples directly as vectors and retrieves "
            "relevant edges first, then reconstructs the local neighborhood context. This "
            "approach shows particular strength on relationship-centric queries. "
            "Both MindMapRAG and EdgeRAG were compared against Microsoft GraphRAG, RAPTOR, "
            "HippoRAG, KGRAG, and GRetriever by Professor Yuki Tanaka group at Stanford. "
            "The comparative study found that EdgeRAG outperforms other systems on "
            "relationship-centric queries while MindMapRAG shows strong performance on "
            "hierarchical concept queries."
        ),
    },
    {
        "id": "doc_019",
        "title": "GRetriever: Graph Reasoning Retriever",
        "content": (
            "GRetriever is a graph-based retrieval system developed by researchers at "
            "Oxford University, led by Dr. Isabelle Fontaine. The system combines graph "
            "neural network encodings with a prize-collecting Steiner tree algorithm to "
            "identify the minimal connected subgraph that best answers a given query. "
            "The core innovation of GRetriever is its use of Steiner tree optimization "
            "to select relevant graph elements. Given a set of seed entities identified "
            "from a query, GRetriever finds the minimum-cost connected subgraph spanning "
            "those seeds, where edge costs are inversely proportional to relevance scores. "
            "This produces a compact, coherent subgraph that includes only the most "
            "relevant connecting evidence. "
            "Dr. Isabelle Fontaine collaborated with Dr. Zhen Liu from the University of "
            "Cambridge on experiments comparing GRetriever with HippoRAG on multi-hop "
            "reasoning tasks. Their joint paper, presented at the KnowledgeGraph Summit 2023, "
            "showed that GRetriever Steiner tree approach consistently produces smaller "
            "but more precise evidence subgraphs compared to HippoRAG associative "
            "traversal approach. "
            "Professor Yuki Tanaka included GRetriever in the GraphRAG benchmark suite, "
            "where it demonstrated competitive performance particularly on multi-hop "
            "reasoning tasks with sparse knowledge graphs. GRetriever uses transformer "
            "architectures for entity embedding and relevance scoring, and graph neural "
            "networks for final answer generation conditioned on the retrieved subgraph."
        ),
    },
    {
        "id": "doc_020",
        "title": "Community Detection in Knowledge Graphs: A Comparative Study",
        "content": (
            "Community detection is a critical preprocessing step in knowledge graph-based "
            "retrieval systems, particularly for global search approaches like Microsoft "
            "GraphRAG. The choice of community detection algorithm directly impacts the "
            "quality of community summaries and, consequently, the quality of answers to "
            "broad, thematic queries. "
            "Dr. Priya Nair from NeuralTech Research Lab conducted a comprehensive comparison "
            "of three major community detection algorithms: Louvain, Leiden, and Label "
            "Propagation. The study used five enterprise knowledge graphs of varying sizes, "
            "from 1,000 to 500,000 nodes, and evaluated each algorithm on modularity score, "
            "community size distribution, internal connectivity, and runtime performance. "
            "The Louvain algorithm, while fast and widely adopted, was found to occasionally "
            "produce internally disconnected communities, which degrades summary quality. "
            "The Leiden algorithm, which adds a refinement phase to guarantee community "
            "connectivity, achieved consistently higher modularity scores and produced "
            "communities that yielded better retrieval quality in downstream RAG evaluations. "
            "Label Propagation was the fastest algorithm but produced the least stable "
            "communities, with results varying across random seeds. "
            "The study, co-authored by Dr. Priya Nair and Professor Yuki Tanaka, "
            "was published in the Journal of Graph Learning and presented at the "
            "KnowledgeGraph Summit 2023. It recommended Leiden as the default community "
            "detection algorithm for Microsoft GraphRAG deployments."
        ),
    },
    {
        "id": "doc_021",
        "title": "NeuralTech Research Lab: Advancing the State of the Art",
        "content": (
            "NeuralTech Research Lab is the dedicated research division of NeuralTech "
            "Corporation, established in 2017 and led by Dr. Priya Nair. The lab operates "
            "with a dual mandate: producing fundamental research on graph machine learning "
            "and knowledge representation, while also maintaining close collaboration with "
            "NeuralTech product engineering teams to ensure research translates into "
            "commercial impact. "
            "The lab research agenda spans four areas: graph neural network architectures, "
            "community detection and graph partitioning, knowledge graph construction from "
            "unstructured text, and evaluation methodology for graph-based AI systems. "
            "Dr. Marcus Chen, NeuralTech CTO, serves as executive sponsor of the lab "
            "and regularly advises on research priorities. "
            "Key collaborations maintained by NeuralTech Research Lab include partnerships "
            "with Professor Yuki Tanaka Knowledge Systems Lab at Stanford University, "
            "Dr. Zhen Liu group at the University of Cambridge, and Dr. Sofia Romero "
            "who joined as a visiting researcher following RAPTOR development. The lab "
            "also has an ongoing exchange program with the AI research team at DataVerse Inc. "
            "Among the lab most impactful contributions is Dr. Nair streaming Louvain "
            "variant for dynamic knowledge graphs, GraphBERT co-developed with Dr. Chen, "
            "and contributions to the GraphRAG benchmark suite. The lab publishes roughly "
            "15 papers per year at top-tier venues including NeurIPS, ICLR, KDD, and "
            "the International Semantic Web Conference."
        ),
    },
    {
        "id": "doc_022",
        "title": "The NeuroGraph Platform: Architecture and Capabilities",
        "content": (
            "NeuroGraph is NeuralTech Corporation flagship enterprise knowledge graph "
            "platform, designed to help organizations build, maintain, and query large-scale "
            "knowledge graphs from heterogeneous data sources. Architected by Dr. Marcus Chen "
            "and the NeuralTech engineering team, NeuroGraph has evolved through four major "
            "versions since its initial release in 2017. "
            "The platform core processing engine, GraphStream, was significantly enhanced "
            "following NeuralTech acquisition of GraphIndex Labs in 2022. The integration "
            "of SpiderIndex technology, developed by Dr. Nadia Petrov and Carlos Mendez at "
            "GraphIndex Labs, reduced query latency by 60 percent for graph traversal "
            "operations on trillion-edge knowledge graphs. "
            "NeuroGraph uses transformer architectures throughout its pipeline: GraphBERT "
            "for entity disambiguation, a fine-tuned sentence transformer for passage "
            "embedding, and a large language model interface for natural language query "
            "processing and answer generation. The platform implements both global search "
            "inspired by Microsoft GraphRAG and local search using entity-centric subgraph "
            "expansion. "
            "Community detection in NeuroGraph uses the Leiden algorithm by default, "
            "following recommendations from NeuralTech Research Lab comparative study "
            "led by Dr. Priya Nair. The platform supports Louvain and Label Propagation "
            "as alternative options. An experimental RAPTOR-style hierarchical indexing "
            "feature, developed in collaboration with visiting researcher Dr. Sofia Romero, "
            "was included in NeuroGraph 4.0 as an optional indexing mode."
        ),
    },
    {
        "id": "doc_023",
        "title": "Competitive Landscape: Knowledge Graph AI Market in 2023",
        "content": (
            "The enterprise knowledge graph AI market experienced explosive growth in 2023, "
            "driven by increasing enterprise adoption of generative AI and the recognition "
            "that language models alone are insufficient for reliable enterprise reasoning. "
            "The market features three dominant commercial players: NeuralTech Corporation, "
            "CogniSystems, and QuantumAI Systems, alongside a rapidly evolving ecosystem "
            "of open-source frameworks and academic systems. "
            "NeuralTech Corporation, led by Dr. Elena Vasquez, holds the strongest position "
            "in pure knowledge graph platforms, with NeuroGraph serving clients in finance, "
            "healthcare, and legal sectors. The company acquisition of GraphIndex Labs "
            "strengthened its infrastructure capabilities significantly. "
            "CogniSystems, despite being the oldest player in the market, has reinvigorated "
            "its product line through the acquisition of TechFlow Inc and the integration "
            "of FlowNLP into CogniGraph 3.0. Dr. Howard Bell has positioned CogniSystems "
            "as the enterprise-safe choice, emphasizing explainability and auditability. "
            "QuantumAI Systems has differentiated itself through the DataVerse Intelligence "
            "partnership, creating a tightly integrated data warehouse plus AI search "
            "offering. Dr. James Okafor has indicated intentions to expand the partnership "
            "with DataVerse into graph-native features in 2024. "
            "Open-source systems including Microsoft GraphRAG, RAPTOR, HippoRAG, KGRAG, "
            "MindMapRAG, EdgeRAG, and GRetriever continue to shape academic and startup "
            "activity in the space, with Professor Yuki Tanaka GraphRAG benchmark "
            "serving as the primary evaluation framework."
        ),
    },
    {
        "id": "doc_024",
        "title": "KGraphCo: AI Startup in the Graph Space",
        "content": (
            "KGraphCo is an early-stage AI startup focused on automated knowledge graph "
            "construction from enterprise documents, headquartered in San Francisco. "
            "Founded in 2021 by Dr. Mei Lin and Thomas Greer, KGraphCo has attracted "
            "attention for its lightweight approach to entity extraction and relation "
            "detection that runs entirely on-premises without requiring cloud LLM APIs. "
            "Dr. Mei Lin, who completed her PhD at Carnegie Mellon University, serves as "
            "CTO of KGraphCo. Thomas Greer, a former product manager at DataVerse Inc, "
            "is the CEO. The company has raised 8 million dollars in seed funding, with "
            "Dr. Elena Vasquez from NeuralTech Corporation as one of the angel investors. "
            "KGraphCo flagship product, GraphBuilder, uses a fine-tuned transformer "
            "architecture specifically optimized for on-premises deployment on standard "
            "CPU hardware, making it accessible to organizations without GPU infrastructure. "
            "GraphBuilder integrates with both NeuroGraph from NeuralTech and CogniGraph "
            "from CogniSystems as a front-end ingestion pipeline. "
            "The company participated in the KnowledgeGraph Summit 2023 as an exhibitor "
            "and gave a lightning talk on Efficient Knowledge Graph Construction for "
            "Resource-Constrained Environments. Dr. Priya Nair from NeuralTech Research "
            "Lab cited KGraphCo approach in her community detection presentation as an "
            "example of the practical constraints that influence algorithm choice in "
            "production deployments."
        ),
    },
    {
        "id": "doc_025",
        "title": "The Future of Graph RAG: Trends and Predictions",
        "content": (
            "As the field of graph-based retrieval augmented generation matures, several "
            "key trends are shaping its trajectory. Industry leaders, academic researchers, "
            "and practitioners who convened at TechSummit 2023 and the KnowledgeGraph "
            "Summit 2023 converged on several predictions for how the field will evolve "
            "over the next three to five years. "
            "Dr. Elena Vasquez from NeuralTech Corporation argued that the next major "
            "breakthrough will come from tighter integration between dynamic knowledge "
            "graphs that update in real time and language model inference. She cited "
            "NeuralTech Research Lab ongoing work on streaming community detection, "
            "led by Dr. Priya Nair, as a key building block for such systems. "
            "Professor Yuki Tanaka from Stanford University emphasized that evaluation "
            "methodology remains an unsolved problem. The GraphRAG benchmark he co-developed "
            "with Dr. Marcus Chen covers a range of query types, but Professor Tanaka "
            "acknowledged that current benchmarks do not adequately capture the full "
            "complexity of enterprise knowledge management tasks. "
            "Dr. James Okafor from QuantumAI Systems predicted that the market would "
            "consolidate around four approaches: global map-reduce search in the "
            "style of Microsoft GraphRAG, local entity-centric search, hybrid dense plus "
            "graph retrieval in the style of QuantumAI Sparse Dense Fusion, and "
            "structured triple-based grounding in the style of KGRAG. "
            "Dr. Zhen Liu from the University of Cambridge and Dr. Isabelle Fontaine from "
            "Oxford University both highlighted the potential of neurologically inspired "
            "systems like HippoRAG and topologically optimal systems like GRetriever to "
            "inform the next generation of enterprise graph RAG platforms."
        ),
    },
]

SAMPLE_QUERIES = [
    "Who founded NeuralTech and what technology do they use?",
    "What is the relationship between QuantumAI and DataVerse?",
    "Which companies are working on transformer architectures?",
    "What events led to the merger between TechFlow and CogniSystems?",
    "Who are the key researchers in the knowledge graph space?",
    "What community detection algorithms are used in graph RAG systems?",
    "How does Microsoft GraphRAG compare to RAPTOR and HippoRAG?",
    "What acquisitions has NeuralTech Corporation made?",
    "Who presented at the KnowledgeGraph Summit 2023?",
    "What is the relationship between Dr. Priya Nair and Stanford University?",
]
