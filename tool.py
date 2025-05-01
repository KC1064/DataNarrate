import re
import os
import json
import nbformat as nbf
import subprocess
from typing import Dict, Any, List, Union
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from groq import Groq
from langchain.chains import RetrievalQA


client = Groq(api_key="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")


embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# ------------------Step 1: Load File-----------------------
def load_document(file_path):
    """Load a document from a file path."""
    print(f"🔄 Loading document: {file_path}")
    
    if file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
        
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
    
    print(f"✅ Document loaded successfully: {len(texts)} text segments")
    return texts

# -------------------Step 2: Split Text---------------------
def split_text(texts, chunk_size=100, chunk_overlap=40):
    """Split text into manageable chunks."""
    print("🔄 Splitting text into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    doc_chunks = text_splitter.create_documents(texts)
    
    print(f"✅ Text split into {len(doc_chunks)} chunks")
    return doc_chunks

# -------------------Step 3: Embeddings---------------------
def create_embeddings(doc_chunks, persist_directory="./chroma_db"):
    """Create and store embeddings in ChromaDB."""
    print("🔄 Creating embeddings...")
    
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    
    print(f"✅ Embeddings stored in ChromaDB at {persist_directory}")
    return vectorstore

# ------------------Step 4: Extract Key Insights-------------------------
def extract_key_insights(doc_chunks):
    """Extract key insights from document chunks using LLM."""
    print("🔄 Extracting key insights from chunks...")
    
    key_insights = []
    
    for i, chunk in enumerate(doc_chunks):
        print(f"  Processing chunk {i+1}/{len(doc_chunks)}", end="\r")
        
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {
                    "role": "user",
                    "content": f"""Extract the key insights from the following text:\n\n"{chunk.page_content}"\n\nReturn only the key insights."""
                }
            ],
            temperature=0.6,
            max_completion_tokens=256,
            top_p=0.95,
            stream=False, 
        )
        
        response_text = completion.choices[0].message.content
        key_insights.append(response_text)
    
    print(f"\n✅ Extracted {len(key_insights)} key insights")
    return key_insights

def save_insights_to_file(key_insights, file_path="key_insights.txt"):
    """Save key insights to a text file."""
    print(f"🔄 Saving insights to {file_path}...")
    
    with open(file_path, 'w') as txt_file:
        for insight in key_insights:
            txt_file.write(f"{insight}\n\n")
    
    print(f"✅ Insights saved to {file_path}")
    return file_path

# -----------------------Step 6: Create Vector Embedding of Text File-----------------------------
def create_insight_embeddings(insight_file_path, persist_directory="./chroma_insight_db"):
    """Create embeddings from the insights file."""
    print(f"🔄 Creating embeddings for insights from {insight_file_path}...")
    
    # Load the insights file
    insight_loader = TextLoader(insight_file_path)
    insight_documents = insight_loader.load()
    insight_texts = [doc.page_content for doc in insight_documents]
    
    # Split the insights into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=40,
        length_function=len,
        is_separator_regex=False,
    )
    insight_chunks = text_splitter.create_documents(insight_texts)
    
    # Create and store the embeddings
    insight_vectorstore = Chroma.from_documents(
        documents=insight_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    insight_vectorstore.persist()
    
    print(f"✅ Key insights embedded and stored in ChromaDB at {persist_directory}")
    return insight_vectorstore

# ----------------------Step 7: User Query & Visualization ----------------------
def determine_chart_type(query):
    """Determine the most appropriate chart type based on the query."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['trend', 'over time', 'growth', 'change', 'timeline']):
        return 'line'
    elif any(word in query_lower for word in ['distribution', 'proportion', 'percentage', 'share', 'breakdown']):
        return 'pie'
    elif any(word in query_lower for word in ['correlation', 'relationship', 'versus', 'vs', 'compare']):
        if 'years' in query_lower or 'months' in query_lower or 'quarters' in query_lower:
            return 'bar'  # Time-based comparisons often work better with bar charts
        else:
            return 'scatter'
    else:
        return 'bar'  # Default to bar chart

def process_user_query(query, persist_directory="./chroma_insight_db"):
    """Process user query and extract structured data."""
    print(f"🔄 Processing query: '{query}'")
    
    # Load the vectorstore
    insight_vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    
    # Create a retriever
    retriever = insight_vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Get relevant documents
    relevant_docs = retriever.get_relevant_documents(query)
    relevant_text = "\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"📄 Found {len(relevant_docs)} relevant documents")
    
    # Run the query through the LLM
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this user query: "{query}". 

Based on the data below, identify relevant numerical comparisons (e.g., revenues, values across years), and return the data in JSON format like this:
{{"2021": 120000, "2022": 150000}}.

Data:\n{relevant_text}"""
            }
        ],
        temperature=0.3,
        max_completion_tokens=300,
    )
    
    response_text = response.choices[0].message.content
    
    # Try to extract JSON data
    try:
        # Look for JSON pattern in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            data = json.loads(response_text)
            
        print(f"✅ Successfully extracted structured data with {len(data)} entries")
        return data
    except json.JSONDecodeError:
        print("❌ Could not parse JSON from LLM response. Here's the raw output:")
        print(response_text)
        # Return a minimal structure to avoid breaking the visualization
        return {"Error": 0, "No data could be extracted": 0}

def create_visualization_notebook(query, data, chart_type=None):
    """Create a Jupyter notebook for visualization."""
    print("🔄 Creating visualization notebook...")
    
    # Determine chart type if not specified
    if chart_type is None:
        chart_type = determine_chart_type(query)
    
    print(f"📊 Using chart type: {chart_type}")
    
    # Create a new notebook
    nb = nbf.v4.new_notebook()
    
    # Add cells to the notebook
    cells = [
        nbf.v4.new_markdown_cell(f"# Visualization for Query: {query}"),
        
        nbf.v4.new_code_cell("""
        # Import necessary libraries
        import json
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        try:
            import plotly.express as px
            from plotly.offline import init_notebook_mode
            init_notebook_mode(connected=True)
            plotly_available = True
        except ImportError:
            plotly_available = False
        from IPython.display import display, HTML
                """),
            
            nbf.v4.new_code_cell(f"""
    # Load the data
    data = {json.dumps(data, indent=2)}
    query = "{query}"
    chart_type = "{chart_type}"

    # Print the data
    print("Query:", query)
    print("Data:", data)
            """),
            
            nbf.v4.new_code_cell("""
    # Convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
    else:
        df = pd.DataFrame(data)
        
    # Sort data if categories look like years or quarters
    if all(isinstance(cat, str) and (cat.isdigit() or cat.startswith('Q')) for cat in df['Category']):
        df = df.sort_values('Category')
        
    display(df)
            """),
            
            nbf.v4.new_markdown_cell("## Matplotlib Visualization"),
            
            nbf.v4.new_code_cell("""
    # Set a professional style
    plt.style.use('ggplot')

    # Create matplotlib visualization
    plt.figure(figsize=(12, 7))

    if chart_type == 'bar':
        ax = plt.bar(df['Category'], df['Value'], color='skyblue', width=0.6)
        # Add value labels on top of bars
        for i, v in enumerate(df['Value']):
            plt.text(i, v + max(df['Value'])*0.02, f'{v:,}', ha='center', fontweight='bold')
            
    elif chart_type == 'line':
        plt.plot(df['Category'], df['Value'], marker='o', linestyle='-', linewidth=2, markersize=8, color='steelblue')
        # Add value labels
        for i, v in enumerate(df['Value']):
            plt.text(i, v + max(df['Value'])*0.02, f'{v:,}', ha='center')
            
    elif chart_type == 'pie':
        plt.pie(df['Value'], labels=df['Category'], autopct='%1.1f%%', startangle=90, 
                shadow=True, explode=[0.05]*len(df['Category']), textprops={'fontsize': 12})
        plt.axis('equal')
        
    elif chart_type == 'scatter':
        plt.scatter(df['Category'], df['Value'], s=100, alpha=0.7, color='steelblue')
        # Add trendline
        z = np.polyfit(range(len(df['Category'])), df['Value'], 1)
        p = np.poly1d(z)
        plt.plot(df['Category'], p(range(len(df['Category']))), "r--", alpha=0.7)
    else:
        plt.bar(df['Category'], df['Value'], color='skyblue')  # Default to bar

    plt.title(query, fontsize=16, pad=20, fontweight='bold')
    plt.xlabel("Category", fontsize=12, labelpad=10)
    plt.ylabel("Value", fontsize=12, labelpad=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45 if len(df) > 5 else 0)
    plt.tight_layout()
    plt.show()
            """),
            
            nbf.v4.new_markdown_cell("## Interactive Plotly Visualization (if available)"),
            
            nbf.v4.new_code_cell("""
    # Create plotly visualization if available
    if plotly_available:
        if chart_type == 'bar':
            fig = px.bar(df, x='Category', y='Value', title=query,
                    color='Value', color_continuous_scale='viridis',
                    text='Value', template='plotly_white')
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                    
        elif chart_type == 'line':
            fig = px.line(df, x='Category', y='Value', title=query, 
                    markers=True, line_shape='linear',
                    template='plotly_white')
            fig.update_traces(marker=dict(size=10))
            
        elif chart_type == 'pie':
            fig = px.pie(df, values='Value', names='Category', title=query,
                    hole=0.4, template='plotly_white')
            
        elif chart_type == 'scatter':
            fig = px.scatter(df, x='Category', y='Value', title=query, 
                        size='Value', template='plotly_white',
                        trendline='ols')
        else:
            fig = px.bar(df, x='Category', y='Value', title=query)

        fig.update_layout(
            title_font_size=20,
            xaxis_title="Category",
            yaxis_title="Value",
            legend_title="Legend",
            font=dict(size=12),
            height=500,
        )
        
        fig.show()
    else:
        print("Plotly is not available. Install it with: pip install plotly")
            """)
        ]
    
    
    nb['cells'] = cells
    
    # Save the notebook
    notebook_path = f"visualization_{query.replace(' ', '_')[:30]}.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"✅ Visualization notebook created: {notebook_path}")
    return notebook_path

def execute_notebook(notebook_path):
    """Execute a Jupyter notebook and generate HTML output."""
    print(f"🔄 Executing notebook: {notebook_path}")
    
    try:
        # Execute the notebook
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "--ExecutePreprocessor.timeout=60", 
             "--inplace", notebook_path],
            check=True
        )
        
        # Convert to HTML for viewing
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "html", notebook_path],
            check=True
        )
        
        html_path = notebook_path.replace('.ipynb', '.html')
        print(f"✅ Notebook executed successfully. HTML output: {html_path}")
        return html_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing notebook: {e}")
        return None


def create_visualization_from_query(query, chart_type=None):
    """Create a visualization from a user query."""
    # Process the query to extract structured data
    data = process_user_query(query)
    
    # Create visualization notebook
    notebook_path = create_visualization_notebook(query, data, chart_type)
    
    # Execute the notebook
    html_path = execute_notebook(notebook_path)
    
    # Return paths
    result = {
        "query": query,
        "chart_type": chart_type or determine_chart_type(query),
        "data": data,
        "notebook_path": notebook_path,
        "html_path": html_path
    }
    
    return result

# Full pipeline function
def run_full_pipeline(document_path):
    """Run the complete document analysis and visualization pipeline."""
    # Step 1: Load document
    texts = load_document(document_path)
    
    # Step 2: Split text
    doc_chunks = split_text(texts)
    
    # Step 3: Create initial embeddings
    create_embeddings(doc_chunks)
    
    # Step 4: Extract key insights
    key_insights = extract_key_insights(doc_chunks)
    
    # Step 5: Save insights to file
    insight_file_path = save_insights_to_file(key_insights)
    
    # Step 6: Create insight embeddings
    create_insight_embeddings(insight_file_path)
    
    print("\n✅ Document processing pipeline complete!")
    print("You can now query the insights with the create_visualization_from_query function.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Document analysis and visualization tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process document command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("document_path", help="Path to the document to process")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create visualization from query")
    viz_parser.add_argument("query", help="Query to visualize")
    viz_parser.add_argument("--chart-type", help="Chart type (bar, line, pie, scatter)")
    
    args = parser.parse_args()
    
    if args.command == "process":
        run_full_pipeline(args.document_path)
    elif args.command == "visualize":
        result = create_visualization_from_query(args.query, args.chart_type)
        print(f"\n✨ Visualization created! ✨")
        print(f"- Notebook: {result['notebook_path']}")
        print(f"- HTML Report: {result['html_path']}")
        print(f"- Chart Type: {result['chart_type']}")
    else:
        parser.print_help()