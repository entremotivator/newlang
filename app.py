import streamlit as st
import json
import openai
from typing import Dict, List, Any, Optional
import uuid
import re
import requests
import time
from datetime import datetime
import base64
import io
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import traceback
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced LangChain Agent Builder Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling with fixed white cards
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 30px;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .component-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
        color: #333;
    }
    .component-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border-left-color: #2196F3;
    }
    .node-card {
        background: #ffffff;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        color: #333;
    }
    .node-card:hover {
        border-color: #4CAF50;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .json-output {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #dee2e6;
        font-family: 'Courier New', monospace;
        color: #333;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
    }
    .execution-log {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #00ff41;
        padding: 20px;
        border-radius: 15px;
        font-family: 'Courier New', monospace;
        max-height: 500px;
        overflow-y: auto;
        border: 2px solid #333;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .status-indicator {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-connected { background: #4CAF50; }
    .status-disconnected { background: #f44336; }
    .status-warning { background: #ff9800; }
    .agent-stats {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 20px rgba(0,114,255,0.3);
    }
    .tool-result {
        background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%);
        border: 2px solid #4CAF50;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        color: #2e7d32;
    }
    .error-message {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        border: 2px solid #f44336;
        color: #c62828;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .property-input {
        width: 100%;
        padding: 10px;
        margin: 8px 0;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        font-size: 14px;
        transition: border-color 0.3s ease;
    }
    .property-input:focus {
        border-color: #4CAF50;
        outline: none;
        box-shadow: 0 0 0 3px rgba(76,175,80,0.1);
    }
    .canvas-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 3px solid #dee2e6;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced data classes with better structure
@dataclass
class NodeConfig:
    id: str
    type: str
    name: str
    x: float
    y: float
    properties: Dict[str, Any]
    inputs: List[str]
    outputs: List[str]
    status: str = "idle"
    last_execution: Optional[datetime] = None
    execution_count: int = 0
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Connection:
    id: str
    from_node: str
    to_node: str
    from_port: str
    to_port: str
    condition: Optional[str] = None
    weight: float = 1.0
    active: bool = True
    
    def to_dict(self):
        return asdict(self)

@dataclass
class AgentFlow:
    nodes: List[NodeConfig]
    connections: List[Connection]
    metadata: Dict[str, Any]
    version: str = "1.0"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    def to_dict(self):
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'connections': [conn.to_dict() for conn in self.connections],
            'metadata': self.metadata,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None
        }

class NodeType(Enum):
    INPUT = "input"
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    ROUTER = "router"
    OUTPUT = "output"
    PROMPT = "prompt"
    PARSER = "parser"
    VALIDATOR = "validator"
    WEBHOOK = "webhook"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    AGGREGATOR = "aggregator"
    TRANSFORMER = "transformer"

# Enhanced session state initialization
def initialize_session_state():
    defaults = {
        'agent_flow': AgentFlow([], [], {}, created_at=datetime.now()),
        'selected_node': None,
        'canvas_data': {'nodes': [], 'connections': []},
        'execution_log': [],
        'agent_running': False,
        'openai_client': None,
        'api_status': 'disconnected',
        'conversation_history': [],
        'current_agent_config': None,
        'saved_agents': {},
        'execution_stats': {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'avg_response_time': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0
        },
        'canvas_zoom': 1.0,
        'canvas_pan': {'x': 0, 'y': 0},
        'debug_mode': False,
        'auto_save': True,
        'theme': 'light'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Enhanced sidebar configuration
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🔐 OpenAI Configuration")

# API Key management with validation
api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key to enable real agent execution",
    placeholder="sk-..."
)

# Model selection
if api_key:
    model_options = [
        "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", 
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]
    default_model = st.sidebar.selectbox(
        "Default Model",
        model_options,
        index=0,
        help="Select the default OpenAI model for LLM nodes"
    )
    
    # API validation with enhanced error handling
    try:
        client = openai.OpenAI(api_key=api_key)
        # Test the API key with a simple request
        models = client.models.list()
        st.session_state.openai_client = client
        st.session_state.api_status = 'connected'
        st.session_state.default_model = default_model
        
        st.sidebar.markdown(
            '<span class="status-indicator status-connected"></span>API Connected ✅',
            unsafe_allow_html=True
        )
        
        # Show available models
        with st.sidebar.expander("Available Models"):
            model_list = [model.id for model in models.data if 'gpt' in model.id][:10]
            for model in model_list:
                st.sidebar.text(f"• {model}")
                
    except Exception as e:
        st.session_state.api_status = 'error'
        st.sidebar.markdown(
            '<span class="status-indicator status-disconnected"></span>API Error ❌',
            unsafe_allow_html=True
        )
        st.sidebar.error(f"API Error: {str(e)[:100]}...")
        logger.error(f"OpenAI API Error: {e}")
else:
    st.sidebar.markdown(
        '<span class="status-indicator status-disconnected"></span>API Not Connected ⚠️',
        unsafe_allow_html=True
    )

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Enhanced component library with more node types
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### 🧩 Enhanced Component Library")

# Expanded components configuration
components_config = {
    "Input": {
        "color": "#4CAF50",
        "icon": "📝",
        "description": "User input capture with validation",
        "category": "Input/Output",
        "properties": {
            "input_type": ["text", "file", "voice", "structured", "multimodal"],
            "validation_regex": r".*",
            "placeholder": "Enter your message...",
            "required": True,
            "max_length": 1000,
            "min_length": 1,
            "allow_empty": False,
            "preprocessing": ["trim", "lowercase", "remove_special_chars"]
        }
    },
    "LLM": {
        "color": "#2196F3",
        "icon": "🤖",
        "description": "Advanced language model processing",
        "category": "AI/ML",
        "properties": {
            "model": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4o-mini", "claude-3", "gemini-pro"],
            "temperature": 0.7,
            "max_tokens": 2000,
            "system_prompt": "You are a helpful AI assistant.",
            "stream": True,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop_sequences": [],
            "response_format": ["text", "json", "structured"],
            "function_calling": False,
            "tools": []
        }
    },
    "Prompt": {
        "color": "#9C27B0",
        "icon": "📋",
        "description": "Advanced prompt template engine",
        "category": "Templates",
        "properties": {
            "template": "Answer the question: {question}",
            "variables": ["question"],
            "template_type": ["simple", "chat", "few_shot", "chain_of_thought"],
            "examples": [],
            "context_window": 4000,
            "dynamic_variables": True,
            "conditional_logic": False
        }
    },
    "Tool": {
        "color": "#FF9800",
        "icon": "🔧",
        "description": "External tools and API integrations",
        "category": "Integration",
        "properties": {
            "tool_type": ["web_search", "calculator", "file_reader", "api_call", "database", "python_code", "web_scraper", "email_sender"],
            "endpoint": "",
            "method": "GET",
            "headers": {},
            "parameters": {},
            "timeout": 30,
            "retry_count": 3,
            "authentication": ["none", "api_key", "oauth", "basic"],
            "rate_limit": 100,
            "cache_results": True
        }
    },
    "Memory": {
        "color": "#607D8B",
        "icon": "🧠",
        "description": "Advanced conversation memory management",
        "category": "Memory",
        "properties": {
            "memory_type": ["conversation_buffer", "conversation_summary", "vector_store", "entity_memory", "knowledge_graph"],
            "max_tokens": 1000,
            "return_messages": True,
            "summary_template": "Summarize the conversation so far.",
            "embedding_model": "text-embedding-ada-002",
            "similarity_threshold": 0.8,
            "max_retrievals": 5
        }
    },
    "Router": {
        "color": "#F44336",
        "icon": "🔀",
        "description": "Intelligent decision routing logic",
        "category": "Logic",
        "properties": {
            "routing_type": ["semantic", "keyword", "model_based", "rule_based", "probability"],
            "conditions": {},
            "default_route": "",
            "confidence_threshold": 0.8,
            "fallback_behavior": "default_route",
            "routing_model": "gpt-3.5-turbo"
        }
    },
    "Parser": {
        "color": "#795548",
        "icon": "🔍",
        "description": "Advanced output parsing and formatting",
        "category": "Processing",
        "properties": {
            "parser_type": ["json", "xml", "regex", "structured", "pydantic", "markdown"],
            "schema": {},
            "format_template": "",
            "error_handling": "strict",
            "validation_rules": {},
            "output_format": "json"
        }
    },
    "Validator": {
        "color": "#E91E63",
        "icon": "✅",
        "description": "Comprehensive input/output validation",
        "category": "Validation",
        "properties": {
            "validation_rules": {},
            "error_message": "Validation failed",
            "strict_mode": True,
            "auto_fix": False,
            "validation_type": ["schema", "regex", "custom", "ai_based"],
            "confidence_threshold": 0.9
        }
    },
    "Webhook": {
        "color": "#00BCD4",
        "icon": "🌐",
        "description": "External webhook integration",
        "category": "Integration",
        "properties": {
            "url": "",
            "method": "POST",
            "headers": {},
            "auth_type": ["none", "bearer", "basic", "api_key"],
            "timeout": 30,
            "retry_policy": "exponential_backoff",
            "payload_template": "{}"
        }
    },
    "Output": {
        "color": "#8BC34A",
        "icon": "📤",
        "description": "Enhanced output formatting and delivery",
        "category": "Input/Output",
        "properties": {
            "format": ["text", "json", "html", "markdown", "pdf", "csv"],
            "template": "",
            "post_process": False,
            "delivery_method": ["display", "file", "email", "webhook"],
            "styling": {}
        }
    },
    "Condition": {
        "color": "#FF5722",
        "icon": "❓",
        "description": "Conditional logic and branching",
        "category": "Logic",
        "properties": {
            "condition_type": ["if_else", "switch", "pattern_match"],
            "conditions": [],
            "default_branch": True,
            "evaluation_mode": ["strict", "fuzzy"]
        }
    },
    "Loop": {
        "color": "#3F51B5",
        "icon": "🔄",
        "description": "Iterative processing loops",
        "category": "Control",
        "properties": {
            "loop_type": ["for", "while", "until"],
            "max_iterations": 10,
            "break_condition": "",
            "iteration_variable": "i"
        }
    },
    "Parallel": {
        "color": "#009688",
        "icon": "⚡",
        "description": "Parallel processing execution",
        "category": "Control",
        "properties": {
            "max_concurrent": 5,
            "timeout": 60,
            "merge_strategy": ["all", "first", "majority"]
        }
    }
}

# Group components by category
categories = {}
for comp_name, comp_info in components_config.items():
    category = comp_info.get("category", "Other")
    if category not in categories:
        categories[category] = []
    categories[category].append((comp_name, comp_info))

# Display components by category
for category, components in categories.items():
    with st.sidebar.expander(f"📁 {category}", expanded=True):
        for comp_name, comp_info in components:
            st.markdown(f"""
            <div class="component-card">
                <strong>{comp_info['icon']} {comp_name}</strong><br>
                <small style="color: #666;">{comp_info['description']}</small>
            </div>
            """, unsafe_allow_html=True)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Enhanced canvas controls
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Canvas Controls")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.canvas_data = {'nodes': [], 'connections': []}
        st.session_state.agent_flow = AgentFlow([], [], {}, created_at=datetime.now())
        st.success("Canvas cleared!")
        st.rerun()

with col2:
    if st.button("💾 Quick Save", use_container_width=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = f"agent_{timestamp}"
        st.session_state.saved_agents[agent_name] = st.session_state.agent_flow
        st.success(f"Saved as {agent_name}!")

# Enhanced save/load functionality
agent_name_input = st.sidebar.text_input("Agent Name", placeholder="my_awesome_agent")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Save As", use_container_width=True) and agent_name_input:
        st.session_state.saved_agents[agent_name_input] = st.session_state.agent_flow
        st.success(f"Agent '{agent_name_input}' saved!")

with col2:
    if st.button("📁 Export", use_container_width=True):
        if st.session_state.agent_flow.nodes:
            export_data = st.session_state.agent_flow.to_dict()
            st.download_button(
                "⬇️ Download JSON",
                json.dumps(export_data, indent=2),
                f"{agent_name_input or 'agent'}.json",
                "application/json"
            )

# Load saved agents
if st.session_state.saved_agents:
    selected_agent = st.sidebar.selectbox(
        "📂 Load Saved Agent",
        [""] + list(st.session_state.saved_agents.keys())
    )
    if selected_agent and st.sidebar.button("📂 Load Agent"):
        st.session_state.agent_flow = st.session_state.saved_agents[selected_agent]
        st.session_state.canvas_data = {
            'nodes': [node.to_dict() for node in st.session_state.agent_flow.nodes],
            'connections': [conn.to_dict() for conn in st.session_state.agent_flow.connections]
        }
        st.success(f"Loaded agent '{selected_agent}'!")
        st.rerun()

# Import functionality
uploaded_file = st.sidebar.file_uploader("📥 Import Agent", type=['json'])
if uploaded_file:
    try:
        import_data = json.load(uploaded_file)
        # Convert back to AgentFlow object
        nodes = [NodeConfig(**node) for node in import_data['nodes']]
        connections = [Connection(**conn) for conn in import_data['connections']]
        st.session_state.agent_flow = AgentFlow(
            nodes=nodes,
            connections=connections,
            metadata=import_data.get('metadata', {}),
            version=import_data.get('version', '1.0')
        )
        st.success("Agent imported successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Import failed: {e}")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Enhanced execution controls
if st.session_state.openai_client:
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown("### 🚀 Agent Execution")
    
    # Enhanced agent stats
    stats = st.session_state.execution_stats
    st.sidebar.markdown(f"""
    <div class="agent-stats">
        <strong>📊 Agent Statistics</strong><br>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
            <div>Runs: {stats['total_runs']}</div>
            <div>Success: {stats['successful_runs']}</div>
            <div>Failed: {stats['failed_runs']}</div>
            <div>Avg Time: {stats['avg_response_time']:.2f}s</div>
            <div>Tokens: {stats['total_tokens_used']}</div>
            <div>Cost: ${stats['total_cost']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced test input with templates
    test_templates = {
        "Simple Question": "What is the capital of France?",
        "Complex Analysis": "Analyze the pros and cons of renewable energy sources and provide a detailed comparison.",
        "Creative Task": "Write a short story about a robot learning to paint.",
        "Technical Query": "Explain how machine learning algorithms work in simple terms.",
        "Custom": ""
    }
    
    template_choice = st.sidebar.selectbox("Test Template", list(test_templates.keys()))
    test_input = st.sidebar.text_area(
        "Test Input",
        value=test_templates[template_choice],
        height=100,
        help="Enter your test input or select a template"
    )
    
    # Execution options
    with st.sidebar.expander("⚙️ Execution Options"):
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        stream_output = st.checkbox("Stream Output", value=True)
        save_conversation = st.checkbox("Save to History", value=True)
        st.session_state.debug_mode = debug_mode
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Execute", use_container_width=True):
            if st.session_state.canvas_data['nodes']:
                st.session_state.agent_running = True
                execute_agent_flow(test_input, debug_mode, stream_output, save_conversation)
            else:
                st.error("No agent flow to execute!")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.agent_running = False
            st.info("Execution stopped!")
    
    # Batch execution
    if st.sidebar.button("🔄 Batch Test", use_container_width=True):
        batch_inputs = [
            "Hello, how are you?",
            "What's the weather like?",
            "Tell me a joke",
            "Explain quantum physics",
            "What's 2+2?"
        ]
        for i, batch_input in enumerate(batch_inputs):
            st.sidebar.write(f"Running test {i+1}/5...")
            execute_agent_flow(batch_input, debug_mode, False, False)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Enhanced execution function with better error handling
def execute_agent_flow(user_input: str, debug_mode: bool = False, stream_output: bool = True, save_conversation: bool = True):
    """Execute the agent flow with enhanced error handling and logging."""
    try:
        start_time = time.time()
        
        if debug_mode:
            st.session_state.execution_log.append(f"🔍 [DEBUG] Starting execution with input: {user_input[:50]}...")
        
        # Validate flow
        if not st.session_state.agent_flow.nodes:
            raise ValueError("No nodes in agent flow")
        
        # Find input and output nodes
        input_nodes = [node for node in st.session_state.agent_flow.nodes if node.type == "input"]
        output_nodes = [node for node in st.session_state.agent_flow.nodes if node.type == "output"]
        
        if not input_nodes:
            raise ValueError("No input node found in flow")
        if not output_nodes:
            raise ValueError("No output node found in flow")
        
        # Execute flow (simplified simulation)
        current_data = {"input": user_input, "context": {}}
        
        # Process each node in topological order (simplified)
        for node in st.session_state.agent_flow.nodes:
            if debug_mode:
                st.session_state.execution_log.append(f"🔄 Processing node: {node.name} ({node.type})")
            
            # Simulate node processing
            if node.type == "llm" and st.session_state.openai_client:
                try:
                    response = st.session_state.openai_client.chat.completions.create(
                        model=node.properties.get("model", "gpt-3.5-turbo"),
                        messages=[
                            {"role": "system", "content": node.properties.get("system_prompt", "You are a helpful assistant.")},
                            {"role": "user", "content": current_data["input"]}
                        ],
                        temperature=node.properties.get("temperature", 0.7),
                        max_tokens=node.properties.get("max_tokens", 1000)
                    )
                    
                    current_data["output"] = response.choices[0].message.content
                    current_data["context"]["tokens_used"] = response.usage.total_tokens
                    
                    if debug_mode:
                        st.session_state.execution_log.append(f"✅ LLM Response: {current_data['output'][:100]}...")
                        
                except Exception as e:
                    st.session_state.execution_log.append(f"❌ LLM Error: {str(e)}")
                    raise
            
            # Update node status
            node.status = "completed"
            node.last_execution = datetime.now()
            node.execution_count += 1
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Update statistics
        st.session_state.execution_stats['total_runs'] += 1
        st.session_state.execution_stats['successful_runs'] += 1
        st.session_state.execution_stats['avg_response_time'] = (
            (st.session_state.execution_stats['avg_response_time'] * (st.session_state.execution_stats['total_runs'] - 1) + execution_time) /
            st.session_state.execution_stats['total_runs']
        )
        
        if 'tokens_used' in current_data.get('context', {}):
            st.session_state.execution_stats['total_tokens_used'] += current_data['context']['tokens_used']
            # Rough cost calculation (adjust based on actual pricing)
            cost_per_token = 0.00002  # Approximate cost
            st.session_state.execution_stats['total_cost'] += current_data['context']['tokens_used'] * cost_per_token
        
        # Log success
        st.session_state.execution_log.append(f"✅ Execution completed in {execution_time:.2f}s")
        st.session_state.execution_log.append(f"📤 Final output: {current_data.get('output', 'No output')}")
        
        # Save to conversation history
        if save_conversation:
            st.session_state.conversation_history.append({
                'timestamp': datetime.now(),
                'input': user_input,
                'output': current_data.get('output', 'No output'),
                'execution_time': execution_time,
                'tokens_used': current_data.get('context', {}).get('tokens_used', 0)
            })
        
        st.session_state.agent_running = False
        
    except Exception as e:
        st.session_state.execution_stats['total_runs'] += 1
        st.session_state.execution_stats['failed_runs'] += 1
        st.session_state.execution_log.append(f"❌ Execution failed: {str(e)}")
        st.session_state.agent_running = False
        logger.error(f"Agent execution failed: {e}")
        logger.error(traceback.format_exc())

# Main header with enhanced styling
st.markdown('<h1 class="main-header">🤖 Advanced LangChain Agent Builder Pro</h1>', unsafe_allow_html=True)

# Enhanced main content area with more tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎨 Visual Builder", 
    "⚙️ Node Settings", 
    "🚀 Execution", 
    "📊 Analytics", 
    "💬 Chat History", 
    "🔧 Advanced"
])

with tab1:
    st.markdown("### 🎨 Enhanced Visual Canvas")
    
    # Canvas toolbar
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        canvas_mode = st.selectbox("Mode", ["Select", "Connect", "Pan"], key="canvas_mode")
    with col2:
        grid_snap = st.checkbox("Grid Snap", value=True)
    with col3:
        show_minimap = st.checkbox("Minimap", value=True)
    with col4:
        auto_layout = st.button("Auto Layout")
    with col5:
        validate_flow = st.button("Validate Flow")
    
    # Enhanced canvas with complete JavaScript
    canvas_html = f"""
    <div class="canvas-container">
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <div id="canvas-container" style="position: relative; background: #ffffff; border-radius: 15px; overflow: hidden;">
                    <canvas id="fabric-canvas" width="1200" height="800"></canvas>
                    {f'<div id="minimap" style="position: absolute; top: 15px; right: 15px; width: 180px; height: 120px; border: 2px solid #ccc; background: rgba(255,255,255,0.9); border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);"></div>' if show_minimap else ''}
                </div>
                <div style="margin-top: 15px; text-align: center; background: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;">
                        <button onclick="addComponent('Input')" style="padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">📝 Input</button>
                        <button onclick="addComponent('LLM')" style="padding: 10px 15px; background: #2196F3; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">🤖 LLM</button>
                        <button onclick="addComponent('Tool')" style="padding: 10px 15px; background: #FF9800; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">🔧 Tool</button>
                        <button onclick="addComponent('Memory')" style="padding: 10px 15px; background: #607D8B; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">🧠 Memory</button>
                        <button onclick="addComponent('Router')" style="padding: 10px 15px; background: #F44336; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">🔀 Router</button>
                        <button onclick="addComponent('Output')" style="padding: 10px 15px; background: #8BC34A; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">📤 Output</button>
                        <button onclick="addComponent('Condition')" style="padding: 10px 15px; background: #FF5722; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">❓ Condition</button>
                        <button onclick="addComponent('Loop')" style="padding: 10px 15px; background: #3F51B5; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: all 0.3s;">🔄 Loop</button>
                    </div>
                    <div style="margin-top: 10px;">
                        <button onclick="toggleConnectionMode()" id="connect-btn" style="padding: 8px 20px; background: #9C27B0; color: white; border: none; border-radius: 8px; cursor: pointer; margin: 0 5px;">🔗 Connect Mode</button>
                        <button onclick="clearCanvas()" style="padding: 8px 20px; background: #f44336; color: white; border: none; border-radius: 8px; cursor: pointer; margin: 0 5px;">🗑️ Clear</button>
                        <button onclick="exportCanvas()" style="padding: 8px 20px; background: #00BCD4; color: white; border: none; border-radius: 8px; cursor: pointer; margin: 0 5px;">💾 Export</button>
                        <button onclick="autoArrange()" style="padding: 8px 20px; background: #795548; color: white; border: none; border-radius: 8px; cursor: pointer; margin: 0 5px;">📐 Auto Arrange</button>
                    </div>
                </div>
            </div>
            <div id="node-properties" style="width: 350px; background: #ffffff; border: 2px solid #e9ecef; border-radius: 15px; padding: 25px; max-height: 800px; overflow-y: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                <h3 style="margin-top: 0; color: #333; border-bottom: 2px solid #e9ecef; padding-bottom: 10px;">🔧 Node Properties</h3>
                <div id="properties-content" style="color: #666;">
                    <div style="text-align: center; padding: 40px 20px;">
                        <div style="font-size: 48px; margin-bottom: 15px;">🎯</div>
                        <p>Select a node to edit its properties</p>
                        <small>Click on any node in the canvas to configure its settings</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
    
    <script>
    // Enhanced Fabric.js canvas with complete functionality
    const canvas = new fabric.Canvas('fabric-canvas', {{
        backgroundColor: '#ffffff',
        selection: true,
        preserveObjectStacking: true,
        enableRetinaScaling: true
    }});
    
    // Enhanced component templates
    const componentTemplates = {json.dumps(components_config)};
    
    // Enhanced state management
    let canvasState = {json.dumps(st.session_state.canvas_data)};
    let selectedNode = null;
    let connectionMode = false;
    let connectionStart = null;
    let nodeCounter = 1;
    let gridSnap = {str(grid_snap).lower()};
    let gridSize = 25;
    let zoomLevel = 1;
    let panMode = false;
    
    // Initialize canvas
    function initializeCanvas() {{
        drawGrid();
        setupEventListeners();
        loadExistingNodes();
        
        // Set canvas mode
        const mode = '{canvas_mode}';
        if (mode === 'Pan') {{
            panMode = true;
            canvas.defaultCursor = 'grab';
        }}
    }}
    
    // Enhanced grid drawing
    function drawGrid() {{
        canvas.getObjects().forEach(obj => {{
            if (obj.isGrid) {{
                canvas.remove(obj);
            }}
        }});
        
        const canvasWidth = canvas.width;
        const canvasHeight = canvas.height;
        
        // Create grid pattern
        for (let i = 0; i <= canvasWidth; i += gridSize) {{
            const line = new fabric.Line([i, 0, i, canvasHeight], {{
                stroke: '#f0f0f0',
                strokeWidth: 1,
                selectable: false,
                evented: false,
                excludeFromExport: true,
                isGrid: true
            }});
            canvas.add(line);
            canvas.sendToBack(line);
        }}
        
        for (let i = 0; i <= canvasHeight; i += gridSize) {{
            const line = new fabric.Line([0, i, canvasWidth, i], {{
                stroke: '#f0f0f0',
                strokeWidth: 1,
                selectable: false,
                evented: false,
                excludeFromExport: true,
                isGrid: true
            }});
            canvas.add(line);
            canvas.sendToBack(line);
        }}
        
        canvas.renderAll();
    }}
    
    // Load existing nodes from session state
    function loadExistingNodes() {{
        if (canvasState.nodes && canvasState.nodes.length > 0) {{
            canvasState.nodes.forEach(nodeData => {{
                createNodeFromData(nodeData);
            }});
        }}
        
        if (canvasState.connections && canvasState.connections.length > 0) {{
            canvasState.connections.forEach(connData => {{
                // Recreate connections (simplified)
                console.log('Loading connection:', connData);
            }});
        }}
    }}
    
    // Create node from existing data
    function createNodeFromData(nodeData) {{
        const template = componentTemplates[nodeData.type];
        if (!template) return;
        
        const group = createNodeGroup(nodeData.type, nodeData.x, nodeData.y, nodeData.id, nodeData.name);
        canvas.add(group);
    }}
    
    // Enhanced node creation with complete functionality
    function createNode(type, x = null, y = null) {{
        const nodeId = type.toLowerCase() + '_' + nodeCounter++;
        const nodeName = type + '_' + nodeId.split('_')[1];
        
        const group = createNodeGroup(type, x, y, nodeId, nodeName);
        canvas.add(group);
        
        // Add to state
        const template = componentTemplates[type];
        canvasState.nodes.push({{
            id: nodeId,
            type: type,
            name: nodeName,
            x: group.left,
            y: group.top,
            properties: JSON.parse(JSON.stringify(template.properties)),
            status: 'idle',
            execution_count: 0
        }});
        
        updateStreamlitState();
        return group;
    }}
    
    // Create node group with enhanced styling
    function createNodeGroup(type, x, y, nodeId, nodeName) {{
        const template = componentTemplates[type];
        
        // Calculate position with grid snap
        const posX = x || Math.random() * 800 + 100;
        const posY = y || Math.random() * 500 + 100;
        const snapX = gridSnap ? Math.round(posX / gridSize) * gridSize : posX;
        const snapY = gridSnap ? Math.round(posY / gridSize) * gridSize : posY;
        
        // Create gradient
        const gradient = new fabric.Gradient({{
            type: 'linear',
            coords: {{ x1: 0, y1: 0, x2: 0, y2: 1 }},
            colorStops: [
                {{ offset: 0, color: template.color }},
                {{ offset: 1, color: fabric.Color.fromHex(template.color).darker(0.2).toHex() }}
            ]
        }});
        
        // Main rectangle
        const rect = new fabric.Rect({{
            width: 160,
            height: 100,
            fill: gradient,
            stroke: '#333',
            strokeWidth: 2,
            rx: 20,
            ry: 20,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 8,
                offsetX: 2,
                offsetY: 2
            }})
        }});
        
        // Icon with better positioning
        const icon = new fabric.Text(template.icon, {{
            fontSize: 28,
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center',
            top: -35,
            left: 0,
            originX: 'center',
            originY: 'center',
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.3)',
                blur: 2,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        // Title with better styling
        const title = new fabric.Text(type, {{
            fontSize: 16,
            fontWeight: 'bold',
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center',
            top: -5,
            left: 0,
            originX: 'center',
            originY: 'center',
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.3)',
                blur: 2,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        // Status indicator with animation
        const statusIndicator = new fabric.Circle({{
            radius: 8,
            fill: '#4CAF50',
            top: -45,
            left: 70,
            originX: 'center',
            originY: 'center',
            shadow: new fabric.Shadow({{
                color: 'rgba(76,175,80,0.5)',
                blur: 4,
                offsetX: 0,
                offsetY: 0
            }})
        }});
        
        // Enhanced input/output ports
        const inputPort = new fabric.Circle({{
            radius: 10,
            fill: '#fff',
            stroke: '#333',
            strokeWidth: 3,
            top: 0,
            left: -80,
            originX: 'center',
            originY: 'center',
            portType: 'input',
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 3,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        const outputPort = new fabric.Circle({{
            radius: 10,
            fill: '#fff',
            stroke: '#333',
            strokeWidth: 3,
            top: 0,
            left: 80,
            originX: 'center',
            originY: 'center',
            portType: 'output',
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 3,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        // Node name label
        const nameLabel = new fabric.Text(nodeName, {{
            fontSize: 12,
            fontFamily: 'Arial',
            fill: 'white',
            textAlign: 'center',
            top: 15,
            left: 0,
            originX: 'center',
            originY: 'center',
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.3)',
                blur: 2,
                offsetX: 1,
                offsetY: 1
            }})
        }});
        
        // Create group with enhanced properties
        const group = new fabric.Group([rect, icon, title, nameLabel, statusIndicator, inputPort, outputPort], {{
            left: snapX,
            top: snapY,
            selectable: true,
            hasControls: true,
            hasBorders: true,
            lockScalingX: true,
            lockScalingY: true,
            nodeId: nodeId || (type.toLowerCase() + '_' + nodeCounter),
            nodeType: type,
            nodeName: nodeName,
            borderColor: template.color,
            cornerColor: template.color,
            cornerStyle: 'circle',
            transparentCorners: false
        }});
        
        // Enhanced event listeners
        group.on('mousedown', function(e) {{
            if (connectionMode) {{
                handleConnectionMode(group);
            }} else if (!panMode) {{
                selectNode(group);
            }}
        }});
        
        group.on('moving', function(e) {{
            if (gridSnap) {{
                const snapX = Math.round(group.left / gridSize) * gridSize;
                const snapY = Math.round(group.top / gridSize) * gridSize;
                group.set({{ left: snapX, top: snapY }});
            }}
            updateConnections();
            updateNodePosition(group);
        }});
        
        group.on('mouseenter', function(e) {{
            if (!connectionMode) {{
                group.set({{
                    shadow: new fabric.Shadow({{
                        color: 'rgba(0,0,0,0.4)',
                        blur: 12,
                        offsetX: 3,
                        offsetY: 3
                    }})
                }});
                canvas.renderAll();
            }}
        }});
        
        group.on('mouseleave', function(e) {{
            if (group !== selectedNode) {{
                group.set({{
                    shadow: new fabric.Shadow({{
                        color: 'rgba(0,0,0,0.2)',
                        blur: 8,
                        offsetX: 2,
                        offsetY: 2
                    }})
                }});
                canvas.renderAll();
            }}
        }});
        
        // Entrance animation
        group.set({{ opacity: 0, scaleX: 0.3, scaleY: 0.3 }});
        group.animate({{ opacity: 1, scaleX: 1, scaleY: 1 }}, {{
            duration: 400,
            easing: fabric.util.ease.easeOutBack,
            onChange: canvas.renderAll.bind(canvas)
        }});
        
        return group;
    }}
    
    // Enhanced connection mode handling
    function toggleConnectionMode() {{
        connectionMode = !connectionMode;
        const btn = document.getElementById('connect-btn');
        
        if (connectionMode) {{
            canvas.defaultCursor = 'crosshair';
            btn.style.background = '#E91E63';
            btn.innerHTML = '🔗 Connecting...';
            btn.style.animation = 'pulse 1s infinite';
        }} else {{
            canvas.defaultCursor = 'default';
            btn.style.background = '#9C27B0';
            btn.innerHTML = '🔗 Connect Mode';
            btn.style.animation = 'none';
            if (connectionStart) {{
                resetConnectionStart();
            }}
        }}
    }}
    
    function handleConnectionMode(group) {{
        if (!connectionStart) {{
            connectionStart = group;
            group.set({{ 
                strokeWidth: 4, 
                stroke: '#ff0000',
                shadow: new fabric.Shadow({{
                    color: 'rgba(255,0,0,0.5)',
                    blur: 10,
                    offsetX: 0,
                    offsetY: 0
                }})
            }});
            canvas.renderAll();
        }} else if (connectionStart !== group) {{
            createConnection(connectionStart, group);
            resetConnectionStart();
        }}
    }}
    
    function resetConnectionStart() {{
        if (connectionStart) {{
            connectionStart.set({{ 
                strokeWidth: 2, 
                stroke: '#333',
                shadow: new fabric.Shadow({{
                    color: 'rgba(0,0,0,0.2)',
                    blur: 8,
                    offsetX: 2,
                    offsetY: 2
                }})
            }});
            connectionStart = null;
            canvas.renderAll();
        }}
    }}
    
    // Enhanced connection creation
    function createConnection(startNode, endNode) {{
        const startPos = startNode.getCenterPoint();
        const endPos = endNode.getCenterPoint();
        
        // Calculate bezier curve control points
        const dx = endPos.x - startPos.x;
        const dy = endPos.y - startPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const controlOffset = Math.min(distance * 0.5, 150);
        
        const pathString = `M ${{startPos.x}} ${{startPos.y}} C ${{startPos.x + controlOffset}} ${{startPos.y}} ${{endPos.x - controlOffset}} ${{endPos.y}} ${{endPos.x}} ${{endPos.y}}`;
        
        const path = new fabric.Path(pathString, {{
            fill: '',
            stroke: '#333',
            strokeWidth: 4,
            selectable: true,
            evented: true,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 4,
                offsetX: 1,
                offsetY: 1
            }}),
            connectionId: 'conn_' + Date.now(),
            fromNode: startNode.nodeId,
            toNode: endNode.nodeId,
            strokeDashArray: [0, 0]
        }});
        
        // Animated arrow head
        const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x);
        const arrowHead = new fabric.Triangle({{
            left: endPos.x - 20,
            top: endPos.y - 8,
            width: 20,
            height: 16,
            fill: '#333',
            selectable: false,
            evented: false,
            angle: (angle * 180 / Math.PI) + 90,
            shadow: new fabric.Shadow({{
                color: 'rgba(0,0,0,0.2)',
                blur: 3,
                offsetX: 1,
                offsetY: 1
            }}),
            connectionId: path.connectionId
        }});
        
        // Add connection animation
        let animationFrame = 0;
        const animateConnection = () => {{
            animationFrame += 0.1;
            const dashOffset = Math.sin(animationFrame) * 10;
            path.set({{ strokeDashArray: [10, 5], strokeDashOffset: dashOffset }});
            canvas.renderAll();
            requestAnimationFrame(animateConnection);
        }};
        
        canvas.add(path);
        canvas.add(arrowHead);
        canvas.sendToBack(path);
        
        // Add to state
        canvasState.connections.push({{
            id: path.connectionId,
            from: startNode.nodeId,
            to: endNode.nodeId,
            fromPort: 'output',
            toPort: 'input'
        }});
        
        updateStreamlitState();
        
        // Start animation
        setTimeout(animateConnection, 100);
    }}
    
    // Node selection with enhanced properties panel
    function selectNode(group) {{
        if (selectedNode) {{
            selectedNode.set({{ 
                strokeWidth: 2,
                shadow: new fabric.Shadow({{
                    color: 'rgba(0,0,0,0.2)',
                    blur: 8,
                    offsetX: 2,
                    offsetY: 2
                }})
            }});
        }}
        
        selectedNode = group;
        group.set({{ 
            strokeWidth: 4,
            shadow: new fabric.Shadow({{
                color: 'rgba(76,175,80,0.6)',
                blur: 15,
                offsetX: 0,
                offsetY: 0
            }})
        }});
        canvas.renderAll();
        
        updatePropertiesPanel(group);
        
        // Notify Streamlit
        window.parent.postMessage({{
            type: 'node_selected',
            nodeId: group.nodeId,
            nodeType: group.nodeType
        }}, '*');
    }}
    
    // Enhanced properties panel
    function updatePropertiesPanel(group) {{
        const nodeData = canvasState.nodes.find(n => n.id === group.nodeId);
        if (!nodeData) return;
        
        const template = componentTemplates[group.nodeType];
        const propertiesHtml = `
            <div style="border-bottom: 2px solid #e9ecef; padding-bottom: 15px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: ${{template.color}};">${{template.icon}} ${{group.nodeType}}</h3>
                <small style="color: #666;">${{template.description}}</small>
            </div>
            
            <div style="margin-bottom: 20px;">
                <label style="display: block; font-weight: bold; margin-bottom: 5px; color: #333;">Node Name:</label>
                <input type="text" value="${{nodeData.name}}" onchange="updateNodeProperty('name', this.value)" 
                       class="property-input" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 8px; font-size: 14px;">
            </div>
            
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 15px 0; color: #333;">Properties</h4>
                ${{generatePropertyInputs(template.properties, nodeData.properties)}}
            </div>
            
            <div style="background: #fff3cd; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ffeaa7;">
                <h4 style="margin: 0 0 10px 0; color: #856404;">Status</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                    <div><strong>Status:</strong> ${{nodeData.status || 'idle'}}</div>
                    <div><strong>Executions:</strong> ${{nodeData.execution_count || 0}}</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <button onclick="duplicateNode()" style="padding: 10px; background: #17a2b8; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;">📋 Duplicate</button>
                <button onclick="deleteNode()" style="padding: 10px; background: #dc3545; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;">🗑️ Delete</button>
            </div>
        `;
        
        document.getElementById('properties-content').innerHTML = propertiesHtml;
    }}
    
    // Generate property inputs with enhanced UI
    function generatePropertyInputs(templateProps, currentProps) {{
        let html = '';
        for (const [key, defaultValue] of Object.entries(templateProps)) {{
            const currentValue = currentProps[key] !== undefined ? currentProps[key] : defaultValue;
            
            html += `<div style="margin-bottom: 15px;">`;
            html += `<label style="display: block; font-weight: bold; margin-bottom: 5px; color: #333; text-transform: capitalize;">${{key.replace(/_/g, ' ')}}:</label>`;
            
            if (Array.isArray(defaultValue)) {{
                // Dropdown for array values
                html += `<select onchange="updateNodeProperty('${{key}}', this.value)" class="property-input" style="width: 100%; padding: 10px; border: 2px solid #e9ecef; border-radius: 8px;">`;
                defaultValue.forEach(option => {{
                    const selected = currentValue === option ? 'selected' : '';
                    html += `<option value="${{option}}" ${{selected}}>${{option}}</option>`;
                }});
                html += `</select>`;
            }} else if (typeof defaultValue === 'boolean') {{
                // Checkbox for boolean values
                const checked = currentValue ? 'checked' : '';
                html += `<label style="display: flex; align-items: center; cursor: pointer;">`;
                html += `<input type="checkbox" ${{checked}} onchange="updateNodeProperty('${{key}}', this.checked)" style="margin-right: 8px; transform: scale(1.2);">`;
                html += `<span>${{currentValue ? 'Enabled' : 'Disabled'}}</span>`;
                html += `</label>`;
            }} else if (typeof defaultValue === 'number') {{
                // Number input
                html += `<input type="number" value="${{currentValue}}" onchange="updateNodeProperty('${{key}}', parseFloat(this.value))" class="property-input" step="0.1">`;
            }} else if (typeof defaultValue === 'object') {{
                // JSON editor for objects
                html += `<textarea onchange="updateNodeProperty('${{key}}', JSON.parse(this.value))" class="property-input" rows="3" placeholder="JSON object">${{JSON.stringify(currentValue, null, 2)}}</textarea>`;
            }} else {{
                // Text input for strings
                const isLongText = key.includes('prompt') || key.includes('template') || key.includes('description');
                if (isLongText) {{
                    html += `<textarea onchange="updateNodeProperty('${{key}}', this.value)" class="property-input" rows="4" placeholder="Enter ${{key.replace(/_/g, ' ')}}">${{currentValue}}</textarea>`;
                }} else {{
                    html += `<input type="text" value="${{currentValue}}" onchange="updateNodeProperty('${{key}}', this.value)" class="property-input" placeholder="Enter ${{key.replace(/_/g, ' ')}}">`;
                }}
            }}
            
            html += `</div>`;
        }}
        return html;
    }}
    
    // Update node property
    function updateNodeProperty(key, value) {{
        if (!selectedNode) return;
        
        const nodeData = canvasState.nodes.find(n => n.id === selectedNode.nodeId);
        if (nodeData) {{
            if (key === 'name') {{
                nodeData.name = value;
                selectedNode.nodeName = value;
                // Update the visual name in the node
                const nameLabel = selectedNode.getObjects().find(obj => obj.text && obj.fontSize === 12);
                if (nameLabel) {{
                    nameLabel.set({{ text: value }});
                    canvas.renderAll();
                }}
            }} else {{
                nodeData.properties[key] = value;
            }}
            updateStreamlitState();
        }}
    }}
    
    // Update node position in state
    function updateNodePosition(group) {{
        const nodeData = canvasState.nodes.find(n => n.id === group.nodeId);
        if (nodeData) {{
            nodeData.x = group.left;
            nodeData.y = group.top;
            updateStreamlitState();
        }}
    }}
    
    // Duplicate node
    function duplicateNode() {{
        if (!selectedNode) return;
        
        const nodeData = canvasState.nodes.find(n => n.id === selectedNode.nodeId);
        if (nodeData) {{
            createNode(nodeData.type, nodeData.x + 50, nodeData.y + 50);
        }}
    }}
    
    // Delete node with animation
    function deleteNode() {{
        if (!selectedNode) return;
        
        // Remove from canvas with animation
        selectedNode.animate({{ opacity: 0, scaleX: 0.1, scaleY: 0.1 }}, {{
            duration: 300,
            easing: fabric.util.ease.easeInBack,
            onChange: canvas.renderAll.bind(canvas),
            onComplete: () => {{
                canvas.remove(selectedNode);
                
                // Remove from state
                canvasState.nodes = canvasState.nodes.filter(n => n.id !== selectedNode.nodeId);
                
                // Remove associated connections
                canvasState.connections = canvasState.connections.filter(c => 
                    c.from !== selectedNode.nodeId && c.to !== selectedNode.nodeId
                );
                
                // Remove connection visuals
                canvas.getObjects().forEach(obj => {{
                    if (obj.connectionId && (obj.fromNode === selectedNode.nodeId || obj.toNode === selectedNode.nodeId)) {{
                        canvas.remove(obj);
                    }}
                }});
                
                selectedNode = null;
                document.getElementById('properties-content').innerHTML = `
                    <div style="text-align: center; padding: 40px 20px; color: #666;">
                        <div style="font-size: 48px; margin-bottom: 15px;">🎯</div>
                        <p>Select a node to edit its properties</p>
                    </div>
                `;
                
                updateStreamlitState();
                canvas.renderAll();
            }}
        }});
    }}
    
    // Clear canvas
    function clearCanvas() {{
        if (confirm('Are you sure you want to clear the entire canvas?')) {{
            canvas.clear();
            canvasState = {{ nodes: [], connections: [] }};
            selectedNode = null;
            drawGrid();
            document.getElementById('properties-content').innerHTML = `
                <div style="text-align: center; padding: 40px 20px; color: #666;">
                    <div style="font-size: 48px; margin-bottom: 15px;">🎯</div>
                    <p>Select a node to edit its properties</p>
                </div>
            `;
            updateStreamlitState();
        }}
    }}
    
    // Auto arrange nodes
    function autoArrange() {{
        const nodes = canvas.getObjects().filter(obj => obj.nodeId);
        if (nodes.length === 0) return;
        
        const cols = Math.ceil(Math.sqrt(nodes.length));
        const spacing = 200;
        const startX = 100;
        const startY = 100;
        
        nodes.forEach((node, index) => {{
            const col = index % cols;
            const row = Math.floor(index / cols);
            const newX = startX + col * spacing;
            const newY = startY + row * spacing;
            
            node.animate({{ left: newX, top: newY }}, {{
                duration: 500,
                easing: fabric.util.ease.easeOutQuart,
                onChange: canvas.renderAll.bind(canvas),
                onComplete: () => {{
                    updateNodePosition(node);
                }}
            }});
        }});
    }}
    
    // Export canvas
    function exportCanvas() {{
        const dataURL = canvas.toDataURL({{
            format: 'png',
            quality: 1,
            multiplier: 2
        }});
        
        const link = document.createElement('a');
        link.download = 'agent_flow.png';
        link.href = dataURL;
        link.click();
    }}
    
    // Update connections when nodes move
    function updateConnections() {{
        // Update connection paths when nodes are moved
        canvas.getObjects().forEach(obj => {{
            if (obj.connectionId && obj.type === 'path') {{
                const fromNode = canvas.getObjects().find(n => n.nodeId === obj.fromNode);
                const toNode = canvas.getObjects().find(n => n.nodeId === obj.toNode);
                
                if (fromNode && toNode) {{
                    const startPos = fromNode.getCenterPoint();
                    const endPos = toNode.getCenterPoint();
                    
                    const dx = endPos.x - startPos.x;
                    const controlOffset = Math.abs(dx) * 0.5;
                    
                    const pathString = `M ${{startPos.x}} ${{startPos.y}} C ${{startPos.x + controlOffset}} ${{startPos.y}} ${{endPos.x - controlOffset}} ${{endPos.y}} ${{endPos.x}} ${{endPos.y}}`;
                    
                    obj.path = fabric.util.parsePath(pathString);
                    
                    // Update arrow head
                    const arrowHead = canvas.getObjects().find(a => a.connectionId === obj.connectionId && a.type === 'triangle');
                    if (arrowHead) {{
                        const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x);
                        arrowHead.set({{
                            left: endPos.x - 20,
                            top: endPos.y - 8,
                            angle: (angle * 180 / Math.PI) + 90
                        }});
                    }}
                }}
            }}
        }});
    }}
    
    // Setup event listeners
    function setupEventListeners() {{
        // Canvas events
        canvas.on('mouse:wheel', function(opt) {{
            const delta = opt.e.deltaY;
            let zoom = canvas.getZoom();
            zoom *= 0.999 ** delta;
            if (zoom > 20) zoom = 20;
            if (zoom < 0.01) zoom = 0.01;
            canvas.zoomToPoint({{ x: opt.e.offsetX, y: opt.e.offsetY }}, zoom);
            opt.e.preventDefault();
            opt.e.stopPropagation();
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                switch(e.key) {{
                    case 's':
                        e.preventDefault();
                        exportCanvas();
                        break;
                    case 'a':
                        e.preventDefault();
                        canvas.getObjects().forEach(obj => {{
                            if (obj.nodeId) canvas.setActiveObject(obj);
                        }});
                        canvas.renderAll();
                        break;
                    case 'Delete':
                    case 'Backspace':
                        if (selectedNode) {{
                            deleteNode();
                        }}
                        break;
                }}
            }}
        }});
    }}
    
    // Update Streamlit state
    function updateStreamlitState() {{
        window.parent.postMessage({{
            type: 'canvas_updated',
            data: canvasState
        }}, '*');
    }}
    
    // Add component function (called by buttons)
    function addComponent(type) {{
        createNode(type);
    }}
    
    // Initialize everything
    initializeCanvas();
    
    // Periodic state sync
    setInterval(updateStreamlitState, 5000);
    
    </script>
    """
    
    # Display the enhanced canvas
    st.components.v1.html(canvas_html, height=900)

with tab2:
    st.markdown("### ⚙️ Enhanced Node Settings")
    
    if st.session_state.selected_node:
        node_data = next((node for node in st.session_state.agent_flow.nodes 
                         if node.id == st.session_state.selected_node), None)
        
        if node_data:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"#### {components_config[node_data.type]['icon']} {node_data.type} Configuration")
                
                # Node name
                new_name = st.text_input("Node Name", value=node_data.name, key="node_name_input")
                if new_name != node_data.name:
                    node_data.name = new_name
                
                # Dynamic property editing based on node type
                template = components_config[node_data.type]
                
                st.markdown("##### Properties")
                for prop_key, default_value in template['properties'].items():
                    current_value = node_data.properties.get(prop_key, default_value)
                    
                    if isinstance(default_value, list):
                        # Dropdown for list values
                        new_value = st.selectbox(
                            prop_key.replace('_', ' ').title(),
                            default_value,
                            index=default_value.index(current_value) if current_value in default_value else 0,
                            key=f"prop_{prop_key}"
                        )
                    elif isinstance(default_value, bool):
                        # Checkbox for boolean values
                        new_value = st.checkbox(
                            prop_key.replace('_', ' ').title(),
                            value=current_value,
                            key=f"prop_{prop_key}"
                        )
                    elif isinstance(default_value, (int, float)):
                        # Number input
                        new_value = st.number_input(
                            prop_key.replace('_', ' ').title(),
                            value=float(current_value),
                            key=f"prop_{prop_key}"
                        )
                    elif isinstance(default_value, dict):
                        # JSON editor for dictionaries
                        new_value = st.text_area(
                            prop_key.replace('_', ' ').title(),
                            value=json.dumps(current_value, indent=2),
                            height=100,
                            key=f"prop_{prop_key}"
                        )
                        try:
                            new_value = json.loads(new_value)
                        except:
                            st.error("Invalid JSON format")
                            new_value = current_value
                    else:
                        # Text input for strings
                        if 'prompt' in prop_key or 'template' in prop_key:
                            new_value = st.text_area(
                                prop_key.replace('_', ' ').title(),
                                value=str(current_value),
                                height=100,
                                key=f"prop_{prop_key}"
                            )
                        else:
                            new_value = st.text_input(
                                prop_key.replace('_', ' ').title(),
                                value=str(current_value),
                                key=f"prop_{prop_key}"
                            )
                    
                    node_data.properties[prop_key] = new_value
            
            with col2:
                st.markdown("#### Node Status")
                
                # Status indicators
                status_color = {
                    'idle': '🔵',
                    'running': '🟡',
                    'completed': '🟢',
                    'error': '🔴'
                }.get(node_data.status, '⚪')
                
                st.markdown(f"**Status:** {status_color} {node_data.status}")
                st.markdown(f"**Executions:** {node_data.execution_count}")
                
                if node_data.last_execution:
                    st.markdown(f"**Last Run:** {node_data.last_execution.strftime('%H:%M:%S')}")
                
                # Node actions
                st.markdown("#### Actions")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔄 Reset", use_container_width=True):
                        node_data.status = 'idle'
                        node_data.execution_count = 0
                        node_data.last_execution = None
                        st.success("Node reset!")
                
                with col_b:
                    if st.button("📋 Duplicate", use_container_width=True):
                        new_node = NodeConfig(
                            id=f"{node_data.type}_{uuid.uuid4().hex[:8]}",
                            type=node_data.type,
                            name=f"{node_data.name}_copy",
                            x=node_data.x + 50,
                            y=node_data.y + 50,
                            properties=node_data.properties.copy(),
                            inputs=node_data.inputs.copy(),
                            outputs=node_data.outputs.copy()
                        )
                        st.session_state.agent_flow.nodes.append(new_node)
                        st.success("Node duplicated!")
                        st.rerun()
                
                if st.button("🗑️ Delete Node", use_container_width=True, type="secondary"):
                    st.session_state.agent_flow.nodes = [
                        n for n in st.session_state.agent_flow.nodes if n.id != node_data.id
                    ]
                    st.session_state.agent_flow.connections = [
                        c for c in st.session_state.agent_flow.connections 
                        if c.from_node != node_data.id and c.to_node != node_data.id
                    ]
                    st.session_state.selected_node = None
                    st.success("Node deleted!")
                    st.rerun()
    else:
        st.info("👆 Select a node from the canvas to configure its settings")
        
        # Show available node types
        st.markdown("#### Available Node Types")
        
        for category, components in categories.items():
            with st.expander(f"📁 {category}"):
                for comp_name, comp_info in components:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"**{comp_info['icon']} {comp_name}**")
                    with col2:
                        st.markdown(f"*{comp_info['description']}*")

with tab3:
    st.markdown("### 🚀 Enhanced Execution Environment")
    
    if st.session_state.openai_client:
        # Execution status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Runs", st.session_state.execution_stats['total_runs'])
        with col2:
            success_rate = (st.session_state.execution_stats['successful_runs'] / 
                          max(st.session_state.execution_stats['total_runs'], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{st.session_state.execution_stats['avg_response_time']:.2f}s")
        
        # Enhanced execution log
        st.markdown("#### 📋 Execution Log")
        
        log_container = st.container()
        with log_container:
            if st.session_state.execution_log:
                log_text = "\n".join(st.session_state.execution_log[-50:])  # Show last 50 entries
                st.markdown(f"""
                <div class="execution-log">
                    {log_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No execution logs yet. Run your agent to see logs here.")
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            st.session_state.execution_log = []
            st.rerun()
        
        # Real-time monitoring
        if st.session_state.agent_running:
            st.markdown("#### 🔄 Real-time Monitoring")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (in real implementation, this would track actual execution)
            import time
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Processing... {i+1}%")
                time.sleep(0.01)
            
            st.success("Execution completed!")
            st.session_state.agent_running = False
    
    else:
        st.warning("⚠️ OpenAI API key required for execution. Please configure it in the sidebar.")

with tab4:
    st.markdown("### 📊 Advanced Analytics Dashboard")
    
    if st.session_state.execution_stats['total_runs'] > 0:
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Executions",
                st.session_state.execution_stats['total_runs'],
                delta=1 if st.session_state.execution_stats['total_runs'] > 0 else 0
            )
        
        with col2:
            st.metric(
                "Success Rate",
                f"{(st.session_state.execution_stats['successful_runs'] / st.session_state.execution_stats['total_runs'] * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Total Tokens",
                f"{st.session_state.execution_stats['total_tokens_used']:,}"
            )
        
        with col4:
            st.metric(
                "Estimated Cost",
                f"${st.session_state.execution_stats['total_cost']:.4f}"
            )
        
        # Charts (mock data for demonstration)
        st.markdown("#### 📈 Performance Trends")
        
        # Create sample data for charts
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Executions': np.random.poisson(5, 30),
            'Success_Rate': np.random.uniform(0.8, 1.0, 30),
            'Response_Time': np.random.uniform(0.5, 3.0, 30)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(performance_data.set_index('Date')[['Executions']])
        
        with col2:
            st.line_chart(performance_data.set_index('Date')[['Response_Time']])
        
        # Node usage statistics
        st.markdown("#### 🔧 Node Usage Statistics")
        
        node_usage = {}
        for node in st.session_state.agent_flow.nodes:
            node_type = node.type
            if node_type not in node_usage:
                node_usage[node_type] = 0
            node_usage[node_type] += node.execution_count
        
        if node_usage:
            usage_df = pd.DataFrame(list(node_usage.items()), columns=['Node Type', 'Executions'])
            st.bar_chart(usage_df.set_index('Node Type'))
        else:
            st.info("No node execution data available yet.")
    
    else:
        st.info("📊 Execute your agent to see analytics data here.")
        
        # Show sample analytics
        st.markdown("#### 📋 Sample Analytics Preview")
        st.image("https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=Analytics+Dashboard+Preview", 
                caption="Analytics will appear here after running your agent")

with tab5:
    st.markdown("### 💬 Conversation History")
    
    if st.session_state.conversation_history:
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search conversations", placeholder="Enter search term...")
        with col2:
            sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])
        
        # Display conversations
        conversations = st.session_state.conversation_history.copy()
        if sort_order == "Oldest First":
            conversations.reverse()
        
        for i, conv in enumerate(conversations):
            if search_term and search_term.lower() not in conv['input'].lower() and search_term.lower() not in conv['output'].lower():
                continue
                
            with st.expander(f"💬 Conversation {len(conversations)-i} - {conv['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**Input:**")
                    st.markdown(f"```\n{conv['input']}\n```")
                
                with col2:
                    st.markdown("**Output:**")
                    st.markdown(f"```\n{conv['output']}\n```")
                
                # Metadata
                st.markdown("**Metadata:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.text(f"⏱️ {conv['execution_time']:.2f}s")
                with col_b:
                    st.text(f"🎯 {conv['tokens_used']} tokens")
                with col_c:
                    if st.button("🔄 Replay", key=f"replay_{i}"):
                        execute_agent_flow(conv['input'])
        
        # Export conversations
        if st.button("📥 Export History"):
            history_json = json.dumps(st.session_state.conversation_history, default=str, indent=2)
            st.download_button(
                "⬇️ Download JSON",
                history_json,
                "conversation_history.json",
                "application/json"
            )
    
    else:
        st.info("💬 No conversation history yet. Execute your agent to start building history.")

with tab6:
    st.markdown("### 🔧 Advanced Configuration")
    
    # System settings
    st.markdown("#### ⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.debug_mode,
            help="Enable detailed logging and debugging information"
        )
        
        st.session_state.auto_save = st.checkbox(
            "💾 Auto Save", 
            value=st.session_state.auto_save,
            help="Automatically save agent configuration"
        )
        
        canvas_theme = st.selectbox(
            "🎨 Canvas Theme",
            ["Light", "Dark", "High Contrast"],
            help="Choose canvas appearance theme"
        )
    
    with col2:
        max_execution_time = st.number_input(
            "⏱️ Max Execution Time (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            help="Maximum time allowed for agent execution"
        )
        
        max_tokens_per_request = st.number_input(
            "🎯 Max Tokens per Request",
            min_value=100,
            max_value=8000,
            value=2000,
            help="Maximum tokens for each LLM request"
        )
        
        retry_attempts = st.number_input(
            "🔄 Retry Attempts",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of retry attempts for failed operations"
        )
    
    # Advanced features
    st.markdown("#### 🚀 Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Validate Agent Flow", use_container_width=True):
            # Validate the current agent flow
            validation_results = []
            
            # Check for input nodes
            input_nodes = [n for n in st.session_state.agent_flow.nodes if n.type == "input"]
            if not input_nodes:
                validation_results.append("❌ No input node found")
            else:
                validation_results.append("✅ Input node present")
            
            # Check for output nodes
            output_nodes = [n for n in st.session_state.agent_flow.nodes if n.type == "output"]
            if not output_nodes:
                validation_results.append("❌ No output node found")
            else:
                validation_results.append("✅ Output node present")
            
            # Check for disconnected nodes
            connected_nodes = set()
            for conn in st.session_state.agent_flow.connections:
                connected_nodes.add(conn.from_node)
                connected_nodes.add(conn.to_node)
            
            disconnected = [n for n in st.session_state.agent_flow.nodes if n.id not in connected_nodes and len(st.session_state.agent_flow.nodes) > 1]
            if disconnected:
                validation_results.append(f"⚠️ {len(disconnected)} disconnected nodes")
            else:
                validation_results.append("✅ All nodes connected")
            
            # Display results
            for result in validation_results:
                if "❌" in result:
                    st.error(result)
                elif "⚠️" in result:
                    st.warning(result)
                else:
                    st.success(result)
    
    with col2:
        if st.button("📊 Generate Flow Report", use_container_width=True):
            # Generate comprehensive flow report
            report = {
                "total_nodes": len(st.session_state.agent_flow.nodes),
                "total_connections": len(st.session_state.agent_flow.connections),
                "node_types": {},
                "complexity_score": 0
            }
            
            for node in st.session_state.agent_flow.nodes:
                if node.type not in report["node_types"]:
                    report["node_types"][node.type] = 0
                report["node_types"][node.type] += 1
            
            # Calculate complexity score
            report["complexity_score"] = (
                len(st.session_state.agent_flow.nodes) * 2 +
                len(st.session_state.agent_flow.connections) * 1.5 +
                len(report["node_types"]) * 3
            )
            
            st.json(report)
    
    # Data management
    st.markdown("#### 💾 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("⚠️ I understand this will delete everything"):
                st.session_state.agent_flow = AgentFlow([], [], {}, created_at=datetime.now())
                st.session_state.canvas_data = {'nodes': [], 'connections': []}
                st.session_state.execution_log = []
                st.session_state.conversation_history = []
                st.session_state.saved_agents = {}
                st.session_state.execution_stats = {
                    'total_runs': 0,
                    'successful_runs': 0,
                    'failed_runs': 0,
                    'avg_response_time': 0,
                    'total_tokens_used': 0,
                    'total_cost': 0.0
                }
                st.success("All data cleared!")
                st.rerun()
    
    with col2:
        if st.button("📤 Backup Data", use_container_width=True):
            backup_data = {
                'agent_flow': st.session_state.agent_flow.to_dict(),
                'execution_stats': st.session_state.execution_stats,
                'conversation_history': st.session_state.conversation_history,
                'saved_agents': {k: v.to_dict() for k, v in st.session_state.saved_agents.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            st.download_button(
                "⬇️ Download Backup",
                json.dumps(backup_data, indent=2),
                f"agent_builder_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with col3:
        backup_file = st.file_uploader("📥 Restore Backup", type=['json'])
        if backup_file:
            try:
                backup_data = json.load(backup_file)
                
                # Restore agent flow
                flow_data = backup_data['agent_flow']
                nodes = [NodeConfig(**node) for node in flow_data['nodes']]
                connections = [Connection(**conn) for conn in flow_data['connections']]
                st.session_state.agent_flow = AgentFlow(
                    nodes=nodes,
                    connections=connections,
                    metadata=flow_data.get('metadata', {}),
                    version=flow_data.get('version', '1.0')
                )
                
                # Restore other data
                st.session_state.execution_stats = backup_data.get('execution_stats', st.session_state.execution_stats)
                st.session_state.conversation_history = backup_data.get('conversation_history', [])
                
                # Restore saved agents
                saved_agents_data = backup_data.get('saved_agents', {})
                st.session_state.saved_agents = {}
                for name, agent_data in saved_agents_data.items():
                    nodes = [NodeConfig(**node) for node in agent_data['nodes']]
                    connections = [Connection(**conn) for conn in agent_data['connections']]
                    st.session_state.saved_agents[name] = AgentFlow(
                        nodes=nodes,
                        connections=connections,
                        metadata=agent_data.get('metadata', {}),
                        version=agent_data.get('version', '1.0')
                    )
                
                st.success("Backup restored successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to restore backup: {e}")

# Footer with enhanced information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🤖 Agent Builder Pro v2.0**
    
    Advanced visual agent builder with enhanced features:
    - Drag & drop interface
    - Real-time execution
    - Advanced analytics
    - Export/import functionality
    """)

with col2:
    st.markdown("""
    **📊 Current Session**
    
    - Nodes: {len(st.session_state.agent_flow.nodes)}
    - Connections: {len(st.session_state.agent_flow.connections)}
    - Executions: {st.session_state.execution_stats['total_runs']}
    - Success Rate: {(st.session_state.execution_stats['successful_runs'] / max(st.session_state.execution_stats['total_runs'], 1) * 100):.1f}%
    """.format(
        len=len,
        st=st
    ))

with col3:
    st.markdown("""
    **🔗 Quick Actions**
    
    - Ctrl+S: Export canvas
    - Ctrl+A: Select all nodes  
    - Delete: Remove selected node
    - Mouse wheel: Zoom canvas
    """)

# Auto-save functionality
if st.session_state.auto_save and st.session_state.agent_flow.nodes:
    # Auto-save every 30 seconds (simplified)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    auto_save_name = f"autosave_{timestamp}"
    if auto_save_name not in st.session_state.saved_agents:
        st.session_state.saved_agents[auto_save_name] = st.session_state.agent_flow
