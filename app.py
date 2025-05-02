import re
import os
import json
import subprocess
from typing import Dict, Any, List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq
import argparse

# Import visualization tool
from tool import create_visualization_from_query, determine_chart_type

# Initialize Groq client
client = Groq(api_key="")


embedding_model = OllamaEmbeddings(model="nomic-embed-text")


def analyze_query_for_visualization(query: str) -> Dict[str, Any]:
    """
    Analyze a user query using an LLM to determine what visualization to create.
    
    Args:
        query: The user query about data visualization
        
    Returns:
        A dictionary with the analyzed query and suggested chart type
    """
    print(f"🔍 Analyzing query: {query}")
    
   
    response = client.chat.completions.create(
        model="gemma2-9b-it",
        messages=[
            {
                "role": "system",
                "content": """You are a data visualization assistant. Analyze the user's query and determine:
1. The core question they're asking about data
2. The most appropriate chart type (bar, line, pie, scatter)
3. Any specific data points or time periods they're interested in

Return your analysis in JSON format:
{
  "refined_query": "The refined data query",
  "chart_type": "bar|line|pie|scatter",
  "data_points": ["specific data points or time periods"],
  "reasoning": "Brief explanation of your recommendation"
}"""
            },
            {
                "role": "user",
                "content": f"Analyze this query for data visualization: {query}"
            }
        ],
        temperature=0.3,
        max_completion_tokens=500,
    )
    
    response_text = response.choices[0].message.content
    
    
    try:
       
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
        else:
            analysis = json.loads(response_text)
            
        print(f"✅ Successfully analyzed query")
        return analysis
    except json.JSONDecodeError:
        print("⚠️ Could not parse JSON from LLM response. Using default analysis.")
        
        return {
            "refined_query": query,
            "chart_type": determine_chart_type(query),
            "data_points": [],
            "reasoning": "Default chart type based on query keywords"
        }

def tool_calling_visualization(query: str) -> Dict[str, Any]:
    """
    Automated tool calling for data visualization based on user query.
    
    Args:
        query: The user query about data visualization
        
    Returns:
        A dictionary with the results of the visualization creation
    """
    print("\n🤖 Starting visualization tool calling process")
    
    # Step 1: Analyze the query using LLM
    analysis = analyze_query_for_visualization(query)
    
    print(f"\n📊 Analysis results:")
    print(f"  - Refined query: {analysis['refined_query']}")
    print(f"  - Recommended chart: {analysis['chart_type']}")
    print(f"  - Reasoning: {analysis['reasoning']}")
    
    # Step 2: Create visualization using the analysis
    result = create_visualization_from_query(
        query=analysis['refined_query'],
        chart_type=analysis['chart_type']
    )
    
    # Step 3: Provide a summary of the action
    print(f"\n✨ Visualization created successfully!")
    print(f"  - Notebook file: {result['notebook_path']}")
    print(f"  - HTML report: {result['html_path']}")
    
    
    try:
        if os.name == 'nt': 
            os.startfile(result['html_path'])
        
    except Exception as e:
        print(f"  - Could not automatically open the HTML file: {e}")
        print(f"  - Please open {result['html_path']} manually in your browser")
    
    return result


def interactive_mode():
    """Run in interactive mode, taking queries from the user."""
    print("🤖 Interactive Data Visualization Assistant")
    print("Enter your queries to generate visualizations. Type 'exit' to quit.")
    
    while True:
        query = input("\n🔍 Enter your query (or 'exit'): ")
        
        if query.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break
            
        try:
            tool_calling_visualization(query)
            
            
            continue_response = input("\nWould you like to try another query? (y/n): ")
            if continue_response.lower() != 'y':
                print("👋 Goodbye!")
                break
                
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Visualization Tool Calling")
    parser.add_argument("--query", help="Query to visualize data")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        tool_calling_visualization(args.query)
    else:
        print("Please provide a query with --query or use --interactive mode")
        parser.print_help()
