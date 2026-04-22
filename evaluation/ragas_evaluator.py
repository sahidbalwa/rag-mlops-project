"""
Module for evaluating the RAG pipeline using Ragas framework.
=== FILE: evaluation/ragas_evaluator.py ===
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, List

# Ragas and Metrics
from datasets import Dataset
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
except ImportError:
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    context_recall = None

# Pipeline integrations
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm_client import LLMClient
from generation.prompt_templates import PromptManager
from mlops.tracking.mlflow_tracker import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RagasEvaluator:
    """
    Evaluates RAG pipeline outputs against a Golden Dataset using Ragas metrics.
    """

    def __init__(self):
        """
        Initializes pipeline components in order to generate answers for evaluation.
        """
        self.retriever = Retriever()
        self.reranker = Reranker(top_n=3)
        self.prompt_mgr = PromptManager()
        self.llm = LLMClient()
        self.tracker = MLflowTracker()

        # Build list of active metrics
        self.metrics = []
        if faithfulness:
            self.metrics.append(faithfulness)
        if answer_relevancy:
            self.metrics.append(answer_relevancy)
        if context_precision:
            self.metrics.append(context_precision)
        if context_recall:
            self.metrics.append(context_recall)

    def evaluate_dataset(self, csv_dataset_path: str) -> Dict[str, float]:
        """
        Runs the full evaluation.

        Args:
            csv_dataset_path (str): Path to CSV containing 'question' and 'ground_truth' columns.

        Returns:
            Dict[str, float]: Average scores across the dataset.
        """
        if evaluate is None:
            logger.error("Ragas package is missing. Cannot perform evaluation.")
            return {}

        if not os.path.exists(csv_dataset_path):
            logger.error(f"Dataset path {csv_dataset_path} does not exist.")
            return {}

        # 1. Load dataset
        golden_df = pd.read_csv(csv_dataset_path)
        if "question" not in golden_df.columns or "ground_truth" not in golden_df.columns:
            logger.error("Dataset must contain 'question' and 'ground_truth' columns.")
            return {}

        logger.info(f"Loaded golden dataset with {len(golden_df)} samples from {csv_dataset_path}.")
        
        # 2. Iterate and collect answers/contexts
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        qa_prompt_template = self.prompt_mgr.get_qa_prompt()

        for idx, row in golden_df.iterrows():
            question = row["question"]
            gt = row["ground_truth"]

            try:
                # Retrieve and Rerank
                raw_docs = self.retriever.retrieve(query=question, top_k=5)
                reranked_docs = self.reranker.rerank(query=question, docs_with_scores=raw_docs)
                
                # Format contexts (Ragas expects a list of context strings per question)
                context_list = [doc.page_content for doc, _ in reranked_docs]
                
                # Combine contexts for Prompt
                joined_context = "\n".join(context_list)
                prompt_str = qa_prompt_template.format(context=joined_context, question=question)
                
                # Generate Answer
                answer_text = self.llm.generate(prompt_str)

                questions.append(question)
                answers.append(answer_text)
                contexts.append(context_list)
                ground_truths.append([gt]) # Ragas expects a list of ground truths per query
            except Exception as e:
                logger.error(f"Error processing row {idx} for question: {question}. Error: {e}")

        if not questions:
            logger.warning("No questions processed successfully.")
            return {}

        # 3. Create HuggingFace Dataset required by Ragas
        eval_data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }
        dataset = Dataset.from_dict(eval_data)

        # 4. Evaluate
        logger.info(f"Running Ragas evaluation with {len(self.metrics)} metrics...")
        try:
            result = evaluate(
                dataset,
                metrics=self.metrics
            )
            scores = result.copy()

            # 5. Log metrics to MLflow
            self.tracker.start_run(run_name="weekly_rag_evaluation")
            self.tracker.log_metrics(scores)
            self.tracker.end_run()

            logger.info(f"Evaluator completed. Ragas Scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Failed during Ragas evaluation phase: {e}")
            return {}
