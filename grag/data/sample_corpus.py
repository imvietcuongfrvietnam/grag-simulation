"""Sample corpus about a fictional AI tech company ecosystem."""

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "NeuralTech Corporation Overview",
        "content": (
            "NeuralTech Corporation was founded in 2015 by Dr. Elena Vasquez and Marcus Chen "
            "in San Francisco. The company specializes in large-scale transformer architectures "
            "and knowledge graph systems. NeuralTech is headquartered in Silicon Valley and "
            "employs over 3,000 researchers and engineers. Dr. Elena Vasquez serves as CEO "
            "while Marcus Chen leads the research division as CTO. NeuralTech's flagship "
            "product, GraphMind, uses a proprietary graph neural network to power enterprise "
            "search. The company raised $500M in Series C funding led by Horizon Ventures in "
            "2022. NeuralTech partners with CloudBase Inc for distributed infrastructure. "
            "Their core technology stack includes transformer models, knowledge graphs, and "
            "vector databases. NeuralTech acquired DataVerse Labs in 2021 to strengthen its "
            "data pipeline capabilities."
        ),
    },
    {
        "id": "doc_002",
        "title": "QuantumAI Research Institute",
        "content": (
            "QuantumAI Research Institute is a non-profit research organization based in "
            "Boston, founded by Professor Alan Reyes in 2018. The institute focuses on "
            "quantum-inspired machine learning algorithms and graph-based retrieval systems. "
            "Professor Alan Reyes previously worked at NeuralTech Corporation before leaving "
            "to establish the institute. QuantumAI collaborates closely with MIT and Stanford "
            "University on joint research projects. The institute has published over 200 papers "
            "on knowledge representation and retrieval-augmented generation. QuantumAI receives "
            "funding from the National Science Foundation and DARPA. Researchers at QuantumAI "
            "include Dr. Priya Nair, who leads the Graph RAG team, and Dr. Tobias Müller, "
            "who specializes in community detection algorithms. QuantumAI recently published "
            "the HopRAG paper, introducing hierarchical graph traversal for multi-hop reasoning."
        ),
    },
    {
        "id": "doc_003",
        "title": "DataVerse Labs Acquisition",
        "content": (
            "DataVerse Labs was acquired by NeuralTech Corporation in March 2021 for $1.2 billion. "
            "DataVerse Labs, founded in 2017 by Sofia Lindqvist in Stockholm, built a distributed "
            "data processing platform called StreamFlow. StreamFlow handles real-time entity "
            "extraction and relation mapping at petabyte scale. After the acquisition, Sofia "
            "Lindqvist joined NeuralTech as VP of Data Infrastructure. The DataVerse technology "
            "was integrated into NeuralTech's GraphMind platform, significantly improving "
            "indexing speeds. DataVerse had previously partnered with CloudBase Inc for "
            "cloud storage solutions. The acquisition strengthened NeuralTech's position "
            "in the enterprise knowledge management market. DataVerse Labs had 150 employees "
            "at the time of acquisition, all of whom transitioned to NeuralTech."
        ),
    },
    {
        "id": "doc_004",
        "title": "TechFlow Systems Profile",
        "content": (
            "TechFlow Systems is an enterprise AI platform company headquartered in Austin, Texas. "
            "Founded in 2014 by James Park and Aisha Mohammed, TechFlow provides workflow "
            "automation powered by natural language processing. TechFlow uses transformer-based "
            "models for document understanding and entity extraction. The company works closely "
            "with CogniSystems on joint AI pipeline projects. TechFlow's platform integrates "
            "with major cloud providers including CloudBase Inc. James Park serves as CEO and "
            "Aisha Mohammed leads product development. TechFlow has 800 enterprise clients "
            "across healthcare, finance, and legal sectors. In 2023, TechFlow merged with "
            "CogniSystems in a deal valued at $800 million, forming the combined entity "
            "TechCogni Group. The merger was driven by complementary technology stacks and "
            "shared clients."
        ),
    },
    {
        "id": "doc_005",
        "title": "CogniSystems Technology Stack",
        "content": (
            "CogniSystems was a knowledge management company based in Seattle, founded in 2016 "
            "by Dr. Rachel Kim. CogniSystems developed the CogniGraph platform, which uses "
            "Leiden community detection to organize enterprise knowledge bases. Dr. Rachel Kim "
            "was previously a researcher at QuantumAI Research Institute before founding "
            "CogniSystems. CogniGraph supports multiple graph retrieval strategies including "
            "personalized PageRank and beam search traversal. CogniSystems merged with "
            "TechFlow Systems in 2023 to form TechCogni Group. Dr. Rachel Kim became Chief "
            "AI Officer of the merged entity. CogniSystems had partnerships with NeuralTech "
            "Corporation for model fine-tuning services. Their research team contributed to "
            "the open-source GraphRAG-Toolkit library."
        ),
    },
    {
        "id": "doc_006",
        "title": "CloudBase Inc Infrastructure",
        "content": (
            "CloudBase Inc is a cloud infrastructure provider based in Seattle, led by CEO "
            "Nathan Brooks. CloudBase provides distributed computing, storage, and networking "
            "services for AI workloads. Major clients include NeuralTech Corporation, TechFlow "
            "Systems, and DataVerse Labs. CloudBase launched GraphCompute in 2022, a managed "
            "graph database service optimized for knowledge graph workloads. GraphCompute "
            "supports billion-scale graphs with sub-second query latency. CloudBase Inc "
            "partners with QuantumAI Research Institute on graph database research. Nathan "
            "Brooks previously co-founded StreamLayer, a real-time data streaming company "
            "acquired by CloudBase in 2020. CloudBase competes with major hyperscalers but "
            "differentiates through specialized graph workload optimization."
        ),
    },
    {
        "id": "doc_007",
        "title": "GraphMind Platform Deep Dive",
        "content": (
            "GraphMind is NeuralTech Corporation's enterprise knowledge graph platform. "
            "Built by a team led by Dr. Elena Vasquez and chief architect Wei Zhang, GraphMind "
            "combines dense vector retrieval with structured graph traversal. The platform "
            "implements the Microsoft GraphRAG architecture for global and local search. "
            "GraphMind uses Leiden community detection to organize knowledge into hierarchical "
            "clusters. For retrieval, GraphMind supports personalized PageRank, beam search, "
            "and hybrid dense-sparse retrieval. The system can index millions of documents "
            "and build knowledge graphs with billions of edges. GraphMind integrates with "
            "StreamFlow from DataVerse Labs for real-time data ingestion. Enterprise clients "
            "use GraphMind for regulatory compliance, competitive intelligence, and research "
            "assistance. Wei Zhang published a paper on GraphMind's architecture at NeurIPS 2023."
        ),
    },
    {
        "id": "doc_008",
        "title": "Dr. Priya Nair Research Profile",
        "content": (
            "Dr. Priya Nair is a senior researcher at QuantumAI Research Institute specializing "
            "in graph-based retrieval-augmented generation. She received her PhD from Stanford "
            "University under Professor Alan Reyes before joining QuantumAI. Dr. Nair's research "
            "focuses on multi-hop reasoning over knowledge graphs and community-aware retrieval. "
            "She is the lead author of the HippoRAG paper, which proposes a hippocampus-inspired "
            "architecture combining personalized PageRank with LLM-based recognition scoring. "
            "Dr. Nair collaborates with researchers at NeuralTech Corporation and CogniSystems. "
            "She serves on the program committee for EMNLP, ACL, and NAACL. Her recent work "
            "includes MindMap-RAG, which builds hierarchical concept maps for structured "
            "reasoning. Dr. Nair received the Best Paper Award at ACL 2023 for her work on "
            "multi-hop graph traversal."
        ),
    },
    {
        "id": "doc_009",
        "title": "RAPTOR Tree Architecture",
        "content": (
            "RAPTOR, developed by researchers at Stanford University including Professor Alan "
            "Reyes and graduate student Kevin Wu, introduces a tree-structured approach to "
            "retrieval-augmented generation. RAPTOR recursively clusters text chunks using "
            "Gaussian Mixture Models and generates abstractive summaries at each level. The "
            "resulting tree captures information at multiple granularities, from fine-grained "
            "passages to high-level document summaries. RAPTOR supports two retrieval modes: "
            "tree traversal, which navigates top-down, and collapsed retrieval, which searches "
            "all levels simultaneously. The approach was shown to outperform flat retrieval on "
            "complex multi-document reasoning tasks. QuantumAI Research Institute adopted RAPTOR "
            "as a component in their HopRAG system. NeuralTech Corporation integrated RAPTOR "
            "into GraphMind for hierarchical document indexing. Kevin Wu joined NeuralTech "
            "after completing his PhD."
        ),
    },
    {
        "id": "doc_010",
        "title": "TechCogni Group Formation",
        "content": (
            "TechCogni Group was formed in January 2023 through the merger of TechFlow Systems "
            "and CogniSystems. The combined company is headquartered in Austin, Texas, with "
            "offices in Seattle. James Park became CEO of TechCogni Group, with Dr. Rachel Kim "
            "as Chief AI Officer and Aisha Mohammed as Chief Product Officer. TechCogni's "
            "combined technology platform merges TechFlow's workflow automation with "
            "CogniSystems' CogniGraph knowledge management system. The merged entity has "
            "over 1,200 enterprise clients and generates $400M in annual recurring revenue. "
            "TechCogni Group raised $300M in post-merger funding led by Horizon Ventures. "
            "The company plans to expand into the European market, with a new office opening "
            "in Berlin. TechCogni continues to use CloudBase Inc for infrastructure and "
            "partners with NeuralTech Corporation for large language model capabilities."
        ),
    },
    {
        "id": "doc_011",
        "title": "HippoRAG Paper Summary",
        "content": (
            "HippoRAG, authored by Dr. Priya Nair at QuantumAI Research Institute, draws "
            "inspiration from the human hippocampus to design a novel RAG architecture. The "
            "system treats the LLM as the neocortex for pattern recognition, the knowledge "
            "graph as the hippocampal index, and the retrieval encoder as the parahippocampal "
            "region. During indexing, HippoRAG extracts OpenIE-style triples from passages "
            "and builds a phrase-level knowledge graph. At query time, seed entities are "
            "identified and Personalized PageRank propagates relevance through the graph. "
            "Passages containing highly-ranked entities are retrieved and re-ranked using "
            "LLM-based recognition scoring. HippoRAG demonstrated superior performance on "
            "multi-hop QA benchmarks including MuSiQue, HotpotQA, and 2WikiMultiHopQA. "
            "The paper was published at NeurIPS 2024. CogniSystems integrated HippoRAG "
            "principles into their CogniGraph platform."
        ),
    },
    {
        "id": "doc_012",
        "title": "Community Detection in Knowledge Graphs",
        "content": (
            "Community detection is a fundamental problem in graph analysis with direct "
            "applications to knowledge graph organization. The Louvain algorithm, developed "
            "by researchers at the Université catholique de Louvain, uses modularity "
            "optimization to find hierarchical community structures efficiently. The Leiden "
            "algorithm, introduced by Traag, Waltman, and van Eck, improves upon Louvain by "
            "guaranteeing well-connected communities. Microsoft's GraphRAG uses Leiden "
            "community detection to partition knowledge graphs into hierarchical clusters "
            "for global summarization. QuantumAI Research Institute published comparative "
            "studies showing Leiden outperforms Louvain on well-connectedness metrics. "
            "Dr. Tobias Müller at QuantumAI developed an adaptive community detection "
            "approach that dynamically adjusts resolution based on graph density. CloudBase "
            "Inc's GraphCompute service supports all major community detection algorithms "
            "as managed operations."
        ),
    },
    {
        "id": "doc_013",
        "title": "Knowledge Graph RAG Approaches",
        "content": (
            "Knowledge Graph RAG (KG-RAG) augments retrieval with structured graph lookups "
            "alongside dense vector search. Unlike pure vector RAG, KG-RAG can answer "
            "structured queries by traversing entity relationships. TechCogni Group's platform "
            "implements KG-RAG using a hybrid retrieval pipeline that combines SPARQL-like "
            "entity lookups with semantic similarity search. The approach first identifies "
            "named entities in the query, retrieves their graph neighborhoods, and combines "
            "this structured context with densely retrieved passages. NeuralTech Corporation "
            "extends this with their GraphMind platform, adding community context from Leiden "
            "partitioning. Research at QuantumAI by Dr. Priya Nair shows that KG-RAG "
            "significantly improves performance on questions requiring relational reasoning. "
            "KG-RAG has been deployed at scale in TechCogni's financial document analysis "
            "product, processing over 10 million documents per day."
        ),
    },
    {
        "id": "doc_014",
        "title": "G-Retriever and Subgraph Selection",
        "content": (
            "G-Retriever, developed at Stanford University, addresses the challenge of "
            "retrieving relevant subgraphs from large knowledge graphs for QA tasks. The "
            "method formulates subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) "
            "problem, where nodes and edges receive prizes based on relevance to the query "
            "and costs based on graph distance. The PCST approximation algorithm finds the "
            "minimal connected subgraph that maximizes prize minus cost. Professor Alan Reyes "
            "at QuantumAI Research Institute contributed to the theoretical foundations of "
            "G-Retriever. The method was evaluated on WebQSP and GraphQA benchmarks, "
            "outperforming baselines on multi-hop questions. NeuralTech Corporation explored "
            "integrating G-Retriever into GraphMind but opted for beam search instead due "
            "to latency constraints. G-Retriever is available as an open-source library "
            "maintained by the Stanford NLP group."
        ),
    },
    {
        "id": "doc_015",
        "title": "Edge-RAG and Relation-Centric Retrieval",
        "content": (
            "Edge-RAG proposes treating graph edges (relations) as first-class retrieval "
            "signals rather than treating the knowledge graph as a node-centric structure. "
            "In Edge-RAG, relations between entities are embedded and indexed independently. "
            "At query time, relevant relations are retrieved first, then the connected "
            "entities and passages are ranked. This approach was developed by researchers "
            "at TechCogni Group building on CogniSystems' earlier CogniGraph work. "
            "Dr. Rachel Kim presented Edge-RAG at EMNLP 2023, where it received strong "
            "interest for its ability to answer relation-specific questions. Edge-RAG "
            "performs particularly well on relationship queries such as 'Who works at X?' "
            "and 'What technology does Y use?' CloudBase Inc's GraphCompute supports "
            "edge-indexed queries as a native operation. NeuralTech evaluated Edge-RAG "
            "and incorporated relation embedding into GraphMind's hybrid retrieval pipeline."
        ),
    },
    {
        "id": "doc_016",
        "title": "MindMap RAG for Structured Reasoning",
        "content": (
            "MindMap-RAG, introduced by Dr. Priya Nair and colleagues at QuantumAI Research "
            "Institute, builds a hierarchical concept map from documents to support "
            "structured multi-hop reasoning. The system first extracts key concepts and "
            "organizes them into a branching mindmap with up to 4 levels of hierarchy. "
            "The root represents the broadest topic, with progressively more specific "
            "concepts at lower levels. Retrieval proceeds hierarchically: the system "
            "first finds the relevant branch, then traverses to the appropriate level "
            "of specificity. MindMap-RAG supports three traversal strategies: hierarchical "
            "top-down, breadth-first for broad coverage, and depth-first for focused "
            "deep dives. The approach was evaluated on the MMLU, MATH, and MedQA benchmarks. "
            "TechCogni Group licensed MindMap-RAG from QuantumAI for use in their healthcare "
            "document analysis product."
        ),
    },
    {
        "id": "doc_017",
        "title": "Horizon Ventures AI Portfolio",
        "content": (
            "Horizon Ventures is a leading venture capital firm with a focus on enterprise AI "
            "and infrastructure. The firm has invested in NeuralTech Corporation, TechCogni "
            "Group, and CloudBase Inc. Horizon Ventures led NeuralTech's $500M Series C in "
            "2022 and TechCogni's $300M post-merger round in 2023. Partner Maya Goldstein "
            "leads the AI infrastructure portfolio at Horizon Ventures. The firm believes "
            "that knowledge graph infrastructure is a foundational layer for enterprise AI. "
            "Horizon Ventures co-invested in DataVerse Labs before its acquisition by "
            "NeuralTech. The firm also funds academic research through the Horizon Research "
            "Fellowship, which supports researchers at QuantumAI Research Institute. "
            "Maya Goldstein sits on the board of NeuralTech Corporation and TechCogni Group."
        ),
    },
    {
        "id": "doc_018",
        "title": "Personalized PageRank in Graph RAG",
        "content": (
            "Personalized PageRank (PPR) is a graph algorithm that computes node importance "
            "relative to a set of seed nodes, making it well-suited for knowledge graph "
            "retrieval. In the context of Graph RAG, PPR allows the system to propagate "
            "relevance from query-matched entities through the graph structure. HippoRAG "
            "by Dr. Priya Nair uses PPR as its core retrieval mechanism. The algorithm "
            "starts with high probability mass on seed entities and iteratively distributes "
            "it to neighbors, with a teleportation factor alpha that balances local vs "
            "global relevance. QuantumAI's research shows PPR outperforms simple BFS for "
            "multi-hop retrieval by naturally discounting long paths. Dr. Tobias Müller "
            "published an analysis of PPR convergence properties on sparse knowledge graphs. "
            "GraphMind by NeuralTech uses PPR as one of several retrieval strategies, "
            "selectable based on query type."
        ),
    },
    {
        "id": "doc_019",
        "title": "Microsoft GraphRAG Architecture",
        "content": (
            "Microsoft's GraphRAG, introduced in the paper 'From Local to Global: A Graph RAG "
            "Approach to Query-Focused Summarization,' addresses the limitations of naive RAG "
            "for global sensemaking queries. The system builds a knowledge graph from the "
            "corpus using LLM-based entity and relation extraction. Leiden community detection "
            "then partitions the graph into hierarchical community structures. For each "
            "community, an LLM generates a summary capturing the key themes and entities. "
            "Global search uses a map-reduce approach: each community summary is rated for "
            "relevance to the query (map), then the top-rated summaries are synthesized into "
            "a final answer (reduce). Local search combines entity-centric graph traversal "
            "with dense text retrieval for focused queries. The approach was evaluated on "
            "datasets about social media discussions and news archives. GraphMind by "
            "NeuralTech Corporation implements this architecture at enterprise scale."
        ),
    },
    {
        "id": "doc_020",
        "title": "GraphRAG-Toolkit Open Source",
        "content": (
            "GraphRAG-Toolkit is an open-source library developed by contributors from "
            "CogniSystems, QuantumAI Research Institute, and independent developers. "
            "The toolkit provides implementations of major Graph RAG algorithms including "
            "Microsoft GraphRAG, HippoRAG, RAPTOR, G-Retriever, and MindMap-RAG. "
            "Dr. Rachel Kim from CogniSystems initiated the project to standardize Graph RAG "
            "evaluation. The toolkit includes a benchmark suite with datasets from MuSiQue, "
            "HotpotQA, WebQSP, and GraphQA. Dr. Tobias Müller contributed the community "
            "detection module supporting Louvain, Leiden, and Label Propagation algorithms. "
            "The library is hosted on GitHub and has over 8,000 stars. CloudBase Inc "
            "sponsors the project's compute infrastructure. NeuralTech Corporation uses "
            "GraphRAG-Toolkit internally for algorithm evaluation and contributed the "
            "hybrid retrieval module integrating dense and sparse search."
        ),
    },
]

SAMPLE_QUERIES = [
    "Who founded NeuralTech Corporation and what is their main product?",
    "What is the relationship between QuantumAI and NeuralTech Corporation?",
    "Which companies are working on transformer architectures for knowledge graphs?",
    "What events led to the formation of TechCogni Group?",
    "Who are the key researchers working on Graph RAG algorithms?",
    "How does HippoRAG use Personalized PageRank for retrieval?",
    "What community detection algorithms are used in Graph RAG systems?",
    "Which companies does Horizon Ventures have investments in?",
    "What is the difference between global search and local search in Microsoft GraphRAG?",
    "How does RAPTOR build its hierarchical tree structure?",
]

GOLD_ANSWERS = {
    SAMPLE_QUERIES[0]: {
        "answer": "NeuralTech Corporation was founded by Dr. Elena Vasquez and Marcus Chen. Their main product is GraphMind.",
        "relevant_docs": ["doc_001", "doc_007"],
        "key_entities": ["Elena Vasquez", "Marcus Chen", "GraphMind", "NeuralTech Corporation"],
    },
    SAMPLE_QUERIES[1]: {
        "answer": "Professor Alan Reyes founded QuantumAI after previously working at NeuralTech. They collaborate on research and QuantumAI researchers work with NeuralTech.",
        "relevant_docs": ["doc_002", "doc_001", "doc_009"],
        "key_entities": ["Alan Reyes", "QuantumAI Research Institute", "NeuralTech Corporation"],
    },
    SAMPLE_QUERIES[3]: {
        "answer": "TechFlow Systems and CogniSystems merged in 2023 due to complementary technology stacks and shared clients, forming TechCogni Group.",
        "relevant_docs": ["doc_004", "doc_005", "doc_010"],
        "key_entities": ["TechFlow Systems", "CogniSystems", "TechCogni Group", "James Park", "Rachel Kim"],
    },
}
