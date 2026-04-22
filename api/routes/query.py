"""
Route for executing the RAG query pipeline.
=== FILE: api/routes/query.py ===
"""

import time
import logging
from fastapi import APIRouter, HTTPException
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse, SourceDocument

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from retrieval.context_builder import ContextBuilder
from generation.llm_client import LLMClient
from generation.prompt_templates import PromptManager
from generation.response_parser import ResponseParser

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Executes the complete RAG pipeline to answer the user's query.
    1. Retrieval -> 2. Reranking -> 3. Context Building -> 4. Generation -> 5. Response Parsing
    """
    start_time = time.time()
    
    try:
        # 1. Retrieve raw documents
        retriever = Retriever()
        raw_docs = retriever.retrieve(query=request.question, top_k=request.top_k)
        
        # 2. Rerank
        reranker = Reranker(top_n=3)
        reranked_docs = reranker.rerank(query=request.question, docs_with_scores=raw_docs)
        
        # 3. Build context
        context_str = ContextBuilder.build_context(reranked_docs)
        
        # 4. Generate Answer
        prompt_mgr = PromptManager()
        qa_prompt_template = prompt_mgr.get_qa_prompt()
        prompt_str = qa_prompt_template.format(context=context_str, question=request.question)
        
        llm = LLMClient()
        raw_answer = llm.generate(prompt_str)
        
        # 5. Parse Response
        parsed = ResponseParser.parse(raw_answer)
        
        # Structure SourceDocuments
        sources = []
        for doc, score in reranked_docs:
            sources.append(SourceDocument(
                source=doc.metadata.get("source", "Unknown"),
                relevance_score=score,
                content_snippet=doc.page_content[:200] + "..."
            ))
            
        process_time = time.time() - start_time
        
        return QueryResponse(
            answer=parsed["answer"],
            has_answer=parsed["has_answer"],
            word_count=parsed["word_count"],
            sources=sources,
            processing_time_sec=process_time
        )
        
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
