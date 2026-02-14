"""
InsightForge - Streamlit Dashboard
AI-Powered Business Intelligence Platform with RAG
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path

# LangChain and RAG imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# PDF processing
import pypdf
import io

# Set page config
st.set_page_config(
    page_title="InsightForge BI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


class InsightForgeRAG:
    """RAG system for Business Intelligence"""
    
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.documents = []
    
    def create_knowledge_base_from_dataframe(self, df):
        """Create knowledge base from DataFrame"""
        documents = []
        
        # Overall summary
        overall = f"""
DATASET OVERVIEW:
Total Records: {len(df):,}
"""
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            overall += f"Date Range: {df['Date'].min()} to {df['Date'].max()}\n"
        
        if 'Sales' in df.columns:
            overall += f"""
Total Sales: ${df['Sales'].sum():,.2f}
Average Sale: ${df['Sales'].mean():,.2f}
Median Sale: ${df['Sales'].median():,.2f}
"""
        
        if 'Customer_Satisfaction' in df.columns:
            overall += f"Average Customer Satisfaction: {df['Customer_Satisfaction'].mean():.2f}/5.0\n"
        
        documents.append(Document(
            page_content=overall,
            metadata={'type': 'overall_summary'}
        ))
        
        # Product summaries
        if 'Product' in df.columns:
            for product in df['Product'].unique():
                pdf = df[df['Product'] == product]
                product_doc = f"""
PRODUCT: {product}
Total Sales: ${pdf['Sales'].sum():,.2f}
Average Sale: ${pdf['Sales'].mean():,.2f}
Transaction Count: {len(pdf):,}
Market Share: {(len(pdf)/len(df)*100):.1f}%
"""
                if 'Customer_Satisfaction' in pdf.columns:
                    product_doc += f"Average Satisfaction: {pdf['Customer_Satisfaction'].mean():.2f}/5.0\n"
                
                documents.append(Document(
                    page_content=product_doc,
                    metadata={'type': 'product', 'product': product}
                ))
        
        # Regional summaries
        if 'Region' in df.columns:
            for region in df['Region'].unique():
                rdf = df[df['Region'] == region]
                region_doc = f"""
REGION: {region}
Total Sales: ${rdf['Sales'].sum():,.2f}
Average Sale: ${rdf['Sales'].mean():,.2f}
Transaction Count: {len(rdf):,}
"""
                if 'Product' in rdf.columns:
                    top_product = rdf.groupby('Product')['Sales'].sum().idxmax()
                    region_doc += f"Top Product: {top_product}\n"
                
                documents.append(Document(
                    page_content=region_doc,
                    metadata={'type': 'region', 'region': region}
                ))
        
        self.documents = documents
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        return len(documents)
    
    def query(self, question, top_k=3):
        """Query the RAG system"""
        if not self.vectorstore:
            return {"answer": "Please upload data first to build the knowledge base.", "sources": []}
        
        docs = self.vectorstore.similarity_search(question, k=top_k)
        context = "\n\n".join([f"SOURCE {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        system_msg = SystemMessage(content="""You are InsightForge, an expert Business Intelligence assistant.

Analyze the provided data context and answer questions with:
- Specific numbers and percentages from the data
- Data-driven insights and patterns
- Actionable, concrete recommendations
- Clear, business-friendly language

Base all answers strictly on the data context provided.""")
        
        user_msg = HumanMessage(content=f"""DATA CONTEXT:
{context}

QUESTION: {question}

Provide a detailed, data-driven answer with specific insights and recommendations:""")
        
        response = self.llm.invoke([system_msg, user_msg])
        
        return {
            'answer': response.content,
            'sources': [{'type': doc.metadata.get('type', 'unknown'), 
                        'preview': doc.page_content[:100]} 
                       for doc in docs]
        }


def load_csv_data(uploaded_file):
    """Load CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def load_excel_data(uploaded_file):
    """Load Excel data"""
    try:
        df = pd.read_excel(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def load_pdf_data(uploaded_file):
    """Extract text from PDF"""
    try:
        pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text, None
    except Exception as e:
        return None, str(e)


def process_data(df):
    """Process and engineer features"""
    df_clean = df.copy()
    
    # Convert date if exists
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean['Year'] = df_clean['Date'].dt.year
        df_clean['Month'] = df_clean['Date'].dt.month
        df_clean['Quarter'] = df_clean['Date'].dt.quarter
    
    # Age groups
    if 'Customer_Age' in df_clean.columns:
        df_clean['Age_Group'] = pd.cut(
            df_clean['Customer_Age'],
            bins=[0, 25, 35, 50, 100],
            labels=['18-25', '26-35', '36-50', '50+']
        )
    
    # Sales categories
    if 'Sales' in df_clean.columns:
        df_clean['Sales_Category'] = pd.cut(
            df_clean['Sales'],
            bins=[0, 300, 600, 1000],
            labels=['Low', 'Medium', 'High']
        )
    
    return df_clean


def create_visualizations(df):
    """Create interactive Plotly visualizations"""
    charts = {}
    
    # Sales trend
    if 'Date' in df.columns and 'Sales' in df.columns:
        monthly = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum().reset_index()
        monthly['Date'] = monthly['Date'].dt.to_timestamp()
        
        fig = px.line(monthly, x='Date', y='Sales', 
                     title='Monthly Sales Trend',
                     labels={'Sales': 'Total Sales ($)', 'Date': 'Month'})
        fig.update_traces(line_color='#2E86AB', line_width=3)
        charts['trend'] = fig
    
    # Product performance
    if 'Product' in df.columns and 'Sales' in df.columns:
        product_sales = df.groupby('Product')['Sales'].sum().reset_index()
        
        fig = px.bar(product_sales, x='Product', y='Sales',
                    title='Sales by Product',
                    labels={'Sales': 'Total Sales ($)'})
        fig.update_traces(marker_color='#F18F01')
        charts['products'] = fig
    
    # Regional performance
    if 'Region' in df.columns and 'Sales' in df.columns:
        regional_sales = df.groupby('Region')['Sales'].sum().reset_index()
        
        fig = px.bar(regional_sales, x='Region', y='Sales',
                    title='Sales by Region',
                    labels={'Sales': 'Total Sales ($)'})
        fig.update_traces(marker_color='#06A77D')
        charts['regions'] = fig
    
    # Satisfaction distribution
    if 'Customer_Satisfaction' in df.columns:
        fig = px.histogram(df, x='Customer_Satisfaction', nbins=25,
                          title='Customer Satisfaction Distribution',
                          labels={'Customer_Satisfaction': 'Satisfaction Score'})
        fig.update_traces(marker_color='#D7263D')
        charts['satisfaction'] = fig
    
    return charts


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 InsightForge</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Business Intelligence with RAG</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        # API Key from secrets (hardcoded - users don't need to enter)
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.success("✅ API key configured (using secure secrets)")
except:
    # Fallback: allow manual entry if secret not set
    groq_api_key = st.text_input(
        "GROQ API Key",
        type="password",
        help="Get your free API key from https://console.groq.com"
    )
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your GROQ API key to continue")
        st.info("💡 GROQ is 100% FREE! Get your key at https://console.groq.com")
        st.stop()
    
    st.success("✅ API key configured")
        
    st.markdown("---")
        
        # File upload
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
    "Choose a file",
    type=['csv', 'xlsx', 'xls', 'pdf'],
    help="Upload CSV, Excel, or PDF files"
        )
        
        st.markdown("---")
        
        # Info
        st.header("ℹ️ About")
        st.info("""
        **InsightForge** is an AI-powered BI platform that uses:
        - 🤖 GROQ LLM (FREE & Fast)
        - 🔍 RAG for data-aware insights
        - 📊 Interactive visualizations
        - 💬 Natural language queries
        """)
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main content area
    if not uploaded_file:
        st.info("👈 Please upload a data file from the sidebar to get started")
        
        # Sample questions
        st.markdown("### 💡 Example Questions You Can Ask")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - Which product is performing best?
            - What are the sales trends?
            - Which region needs attention?
            - How can we increase revenue?
            """)
        
        with col2:
            st.markdown("""
            - What customer segments are most valuable?
            - What are our growth opportunities?
            - What strategies would you recommend?
            - What risks should we monitor?
            """)
        
        return
    
    # Process uploaded file
    with st.spinner("📊 Processing your data..."):
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            df, error = load_csv_data(uploaded_file)
        elif file_type in ['xlsx', 'xls']:
            df, error = load_excel_data(uploaded_file)
        elif file_type == 'pdf':
            text, error = load_pdf_data(uploaded_file)
            if text and not error:
                st.info("📄 PDF text extracted. For best results, use CSV/Excel for structured data analysis.")
                st.text_area("Extracted Text", text[:1000] + "...", height=200)
            st.stop()
        else:
            st.error("Unsupported file type")
            st.stop()
        
        if error:
            st.error(f"Error loading file: {error}")
            st.stop()
        
        # Process data
        df = process_data(df)
        st.session_state.df = df
        
        # Initialize RAG
        if st.session_state.rag is None:
            st.session_state.rag = InsightForgeRAG(groq_api_key)
        
        # Build knowledge base
        num_docs = st.session_state.rag.create_knowledge_base_from_dataframe(df)
        
    st.success(f"✅ Data loaded! {len(df):,} records • {num_docs} knowledge documents created")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Assistant", "📈 Data Explorer"])
    
    # Tab 1: Dashboard
    with tab1:
        st.header("Business Metrics Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records",
                f"{len(df):,}",
                help="Total number of transactions"
            )
        
        with col2:
            if 'Sales' in df.columns:
                st.metric(
                    "Total Revenue",
                    f"${df['Sales'].sum():,.0f}",
                    help="Sum of all sales"
                )
        
        with col3:
            if 'Sales' in df.columns:
                st.metric(
                    "Average Sale",
                    f"${df['Sales'].mean():,.2f}",
                    help="Mean transaction value"
                )
        
        with col4:
            if 'Customer_Satisfaction' in df.columns:
                st.metric(
                    "Avg Satisfaction",
                    f"{df['Customer_Satisfaction'].mean():.2f}/5.0",
                    help="Customer satisfaction score"
                )
        
        st.markdown("---")
        
        # Visualizations
        charts = create_visualizations(df)
        
        if charts:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'trend' in charts:
                    st.plotly_chart(charts['trend'], use_container_width=True)
                if 'products' in charts:
                    st.plotly_chart(charts['products'], use_container_width=True)
            
            with col2:
                if 'regions' in charts:
                    st.plotly_chart(charts['regions'], use_container_width=True)
                if 'satisfaction' in charts:
                    st.plotly_chart(charts['satisfaction'], use_container_width=True)
    
    # Tab 2: AI Assistant
    with tab2:
        st.header("💬 AI-Powered Business Assistant")
        st.markdown("Ask questions about your data in natural language!")
        
        # Chat interface
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., Which product has the highest sales?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if ask_button and user_question:
            with st.spinner("🤔 Analyzing your data..."):
                result = st.session_state.rag.query(user_question)
                
                # Add to history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': result['answer'],
                    'timestamp': datetime.now()
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    st.markdown(f"**❓ Question {len(st.session_state.chat_history)-i}:**")
                    st.info(chat['question'])
                    
                    st.markdown("**💡 Answer:**")
                    st.success(chat['answer'])
                    
                    st.caption(f"⏰ {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.markdown("---")
    
    # Tab 3: Data Explorer
    with tab3:
        st.header("📈 Data Explorer")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
