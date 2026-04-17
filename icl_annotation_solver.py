"""
FlagOS Track 3: LLM Automatic Data Annotation in Long-Context Scenarios
=====================================================================
ICL (In-Context Learning) based solution for automatic data annotation
using Qwen3-4B model.

This script is designed to run on Kaggle with GPU acceleration.
Approach: Multi-strategy ICL with chain-of-thought, self-consistency,
and long-context window optimization.
"""

import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Model
    MODEL_NAME = "Qwen/Qwen3-4B"  # or FlagOS-specific variant
    MAX_LENGTH = 32768  # Qwen3 native max context
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3  # Low temp for annotation consistency
    TOP_P = 0.9
    TOP_K = 50

    # ICL Settings
    NUM_SHOT_EXAMPLES = 5  # Number of few-shot examples
    CHAIN_OF_THOUGHT = True  # Enable CoT reasoning
    SELF_CONSISTENCY_RUNS = 3  # Multiple runs for self-consistency

    # Data
    INPUT_DIR = "/kaggle/input/track-3-llm-automatic-data-annotation-in-long-context-scenarios"
    OUTPUT_DIR = "/kaggle/working"
    BATCH_SIZE = 1  # Process one at a time for long context

    # Optimization
    USE_4BIT = True  # 4-bit quantization for memory efficiency
    FLASH_ATTENTION = True


# ============================================================
# ICL PROMPT ENGINEERING
# ============================================================
class ICLPromptBuilder:
    """Builds optimized ICL prompts for data annotation tasks."""

    def __init__(self, config: Config):
        self.config = config
        self.system_prompt = (
            "You are an expert data annotator specializing in long-context understanding. "
            "Your task is to carefully read the provided context and annotate the target "
            "text according to the given label schema. Think step-by-step to ensure accuracy.\n\n"
            "Instructions:\n"
            "1. Read the full context carefully\n"
            "2. Identify the relevant information for annotation\n"
            "3. Consider the label definitions provided\n"
            "4. Apply the annotation with confidence\n"
            "5. If uncertain, choose the most probable label"
        )

    def build_annotation_prompt(
        self,
        context: str,
        target_text: str,
        label_schema: List[str],
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build a complete ICL prompt for annotation."""
        prompt_parts = []

        # System instruction
        prompt_parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")

        # User message with ICL examples
        prompt_parts.append("<|im_start|>user")

        # Label schema
        schema_str = ", ".join(label_schema)
        prompt_parts.append(
            f"Label Schema: [{schema_str}]\n\n"
            "Here are some annotated examples:\n"
        )

        # Add few-shot examples if provided
        if examples:
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(
                    f"\nExample {i}:\n"
                    f"Context: {ex['context']}\n"
                    f"Target Text: {ex['target']}\n"
                )
                if self.config.CHAIN_OF_THOUGHT:
                    prompt_parts.append(
                        f"Reasoning: {ex.get('reasoning', 'N/A')}\n"
                    )
                prompt_parts.append(f"Annotation: {ex['label']}\n")

        # The actual task
        prompt_parts.append(
            f"\nNow annotate the following:\n"
            f"Context: {context}\n"
            f"Target Text: {target_text}\n"
        )

        if self.config.CHAIN_OF_THOUGHT:
            prompt_parts.append(
                "Please think step by step and provide your reasoning, "
                "then give the final annotation.\n"
            )

        prompt_parts.append("<|im_end|>")

        # Assistant response prefix
        prompt_parts.append("<|im_start|>assistant")

        return "\n".join(prompt_parts)

    def build_long_context_prompt(
        self,
        documents: List[str],
        query: str,
        task_description: str,
        examples: Optional[List[Dict]] = None,
    ) -> str:
        """Build a prompt optimized for long-context scenarios."""
        prompt_parts = []

        prompt_parts.append(
            f"<|im_start|>system\n"
            f"{self.system_prompt}\n\n"
            f"Task: {task_description}<|im_end|>"
        )

        prompt_parts.append("<|im_start|>user")

        # Concatenate documents for long-context
        combined_docs = "\n\n---\n\n".join(documents)
        # Truncate if too long (keep last N tokens worth)
        max_doc_chars = 20000  # Conservative limit
        if len(combined_docs) > max_doc_chars:
            combined_docs = combined_docs[-max_doc_chars:]
            combined_docs = "...[truncated]...\n" + combined_docs

        prompt_parts.append(
            f"## Reference Documents\n{combined_docs}\n\n"
            f"## Query\n{query}\n"
        )

        if self.config.CHAIN_OF_THOUGHT:
            prompt_parts.append(
                "\nPlease analyze the reference documents carefully, "
                "reason step by step, and provide your annotation."
            )

        prompt_parts.append("<|im_end|>")
        prompt_parts.append("<|im_start|>assistant")

        return "\n".join(prompt_parts)


# ============================================================
# ANNOTATION ENGINE
# ============================================================
class ICLEngine:
    """Main engine for ICL-based data annotation."""

    def __init__(self, config: Config):
        self.config = config
        self.prompt_builder = ICLPromptBuilder(config)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load Qwen3-4B with quantization for efficiency."""
        print(f"Loading model: {self.config.MODEL_NAME}")

        # Quantization config for memory efficiency
        bnb_config = None
        if self.config.USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            **model_kwargs,
        )
        self.model.eval()

        print("Model loaded successfully!")
        print(f"Device: {self.model.device}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def generate_annotation(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a single annotation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.MAX_LENGTH,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                temperature=temperature or self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                top_k=self.config.TOP_K,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        return response.strip()

    def extract_label(self, response: str, label_schema: List[str]) -> str:
        """Extract the annotation label from model response."""
        response_lower = response.lower()

        # Try to find explicit label markers
        for label in label_schema:
            if label.lower() in response_lower:
                return label

        # Try to find "Annotation:" or "Answer:" patterns
        patterns = [
            r"annotation:\s*(.+?)(?:\n|$)",
            r"answer:\s*(.+?)(?:\n|$)",
            r"label:\s*(.+?)(?:\n|$)",
            r"final[^:]*:\s*(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                candidate = match.group(1).strip()
                for label in label_schema:
                    if label.lower() in candidate:
                        return label

        # Fallback: return first line or most common label word
        first_line = response.split("\n")[0].strip()
        for label in label_schema:
            if label.lower() in first_line.lower():
                return label

        # Last resort: return the last meaningful token
        return response.split()[-1] if response else label_schema[0]

    def annotate_with_self_consistency(
        self,
        prompt: str,
        label_schema: List[str],
        num_runs: int = 3,
    ) -> Tuple[str, Dict]:
        """Use self-consistency across multiple generation runs."""
        label_counts = {}
        reasoning_samples = []

        for run in range(num_runs):
            temp = self.config.TEMPERATURE + run * 0.15  # Vary temperature
            response = self.generate_annotation(prompt, temperature=temp)
            label = self.extract_label(response, label_schema)
            label_counts[label] = label_counts.get(label, 0) + 1
            reasoning_samples.append(response)

        # Majority vote
        best_label = max(label_counts, key=label_counts.get)

        return best_label, {
            "label_counts": label_counts,
            "reasoning_samples": reasoning_samples,
            "confidence": label_counts[best_label] / num_runs,
        }

    def process_long_context_sample(
        self,
        row: Dict,
        label_schema: List[str],
        examples: Optional[List[Dict]] = None,
    ) -> Dict:
        """Process a single long-context annotation sample."""
        # Adapt based on actual data format
        context = row.get("context", row.get("documents", row.get("text", "")))
        query = row.get("query", row.get("question", row.get("target", "")))
        task_desc = row.get("task_description", "Annotate the given text.")

        if isinstance(context, str) and context.startswith("["):
            try:
                context = json.loads(context)
                if isinstance(context, list):
                    context = "\n\n".join(str(c) for c in context)
            except json.JSONDecodeError:
                pass

        prompt = self.prompt_builder.build_long_context_prompt(
            documents=[str(context)] if not isinstance(context, list) else [str(c) for c in context],
            query=str(query),
            task_description=task_desc,
            examples=examples,
        )

        if self.config.SELF_CONSISTENCY_RUNS > 1:
            label, meta = self.annotate_with_self_consistency(
                prompt,
                label_schema,
                num_runs=self.config.SELF_CONSISTENCY_RUNS,
            )
        else:
            response = self.generate_annotation(prompt)
            label = self.extract_label(response, label_schema)
            meta = {"reasoning": response, "confidence": 1.0}

        return {
            "label": label,
            "confidence": meta.get("confidence", 1.0),
            "reasoning": meta.get("reasoning_samples", [meta.get("reasoning", "")])[0],
        }


# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================
class DataLoader:
    """Handles loading and preprocessing competition data."""

    def __init__(self, config: Config):
        self.config = config
        self.label_schema = []
        self.train_examples = []
        self.test_samples = []

    def discover_data_files(self):
        """Discover available data files in the input directory."""
        print(f"Scanning data directory: {self.config.INPUT_DIR}")
        if not os.path.exists(self.config.INPUT_DIR):
            print(f"WARNING: Input directory not found!")
            return

        for root, dirs, files in os.walk(self.config.INPUT_DIR):
            for f in files:
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                print(f"  Found: {f} ({fsize / 1024:.1f} KB)")

    def load_train_data(self) -> pd.DataFrame:
        """Load training/annotation data."""
        train_path = os.path.join(
            self.config.INPUT_DIR, "train.csv"
        )
        if not os.path.exists(train_path):
            # Try alternate names
            for alt in ["train.json", "train.parquet", "annotations.csv",
                        "train_data.csv", "annotated_data.csv"]:
                alt_path = os.path.join(self.config.INPUT_DIR, alt)
                if os.path.exists(alt_path):
                    train_path = alt_path
                    break

        if os.path.exists(train_path):
            if train_path.endswith(".json"):
                df = pd.read_json(train_path)
            elif train_path.endswith(".parquet"):
                df = pd.read_parquet(train_path)
            else:
                df = pd.read_csv(train_path)
            print(f"Loaded training data: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df.head())
            return df

        print("No training data found. Will use zero-shot approach.")
        return pd.DataFrame()

    def load_test_data(self) -> pd.DataFrame:
        """Load test data for prediction."""
        test_path = os.path.join(
            self.config.INPUT_DIR, "test.csv"
        )
        if not os.path.exists(test_path):
            for alt in ["test.json", "test.parquet", "submission.csv",
                        "test_data.csv", "evaluate.csv"]:
                alt_path = os.path.join(self.config.INPUT_DIR, alt)
                if os.path.exists(alt_path):
                    test_path = alt_path
                    break

        if os.path.exists(test_path):
            if test_path.endswith(".json"):
                df = pd.read_json(test_path)
            elif test_path.endswith(".parquet"):
                df = pd.read_parquet(test_path)
            else:
                df = pd.read_csv(test_path)
            print(f"Loaded test data: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df.head())
            return df

        raise FileNotFoundError("No test data found!")

    def load_sample_submission(self) -> pd.DataFrame:
        """Load sample submission to understand format."""
        sample_path = os.path.join(
            self.config.INPUT_DIR, "sample_submission.csv"
        )
        if not os.path.exists(sample_path):
            for alt in ["sample_submission.csv", "sample.csv", "submission_format.csv"]:
                alt_path = os.path.join(self.config.INPUT_DIR, alt)
                if os.path.exists(alt_path):
                    sample_path = alt_path
                    break

        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            print(f"Sample submission format: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(df.head())
            return df

        return pd.DataFrame(columns=["ID", "Predicted"])

    def extract_label_schema(self, train_df: pd.DataFrame) -> List[str]:
        """Extract label schema from training data."""
        if train_df.empty:
            return ["positive", "negative", "neutral"]  # Default

        label_col = None
        for col in ["label", "annotation", "category", "class", "target"]:
            if col in train_df.columns:
                label_col = col
                break

        if label_col:
            labels = train_df[label_col].unique().tolist()
            print(f"Discovered label schema: {labels}")
            return labels

        return ["positive", "negative", "neutral"]

    def prepare_icl_examples(
        self, train_df: pd.DataFrame, num_examples: int = 5
    ) -> List[Dict]:
        """Prepare ICL few-shot examples from training data."""
        if train_df.empty:
            return []

        examples = []
        # Get diverse examples by sampling across labels
        label_col = None
        for col in ["label", "annotation", "category", "class", "target"]:
            if col in train_df.columns:
                label_col = col
                break

        context_col = None
        for col in ["context", "documents", "text", "input", "passage"]:
            if col in train_df.columns:
                context_col = col
                break

        query_col = None
        for col in ["query", "question", "target_text", "target"]:
            if col in train_df.columns:
                query_col = col
                break

        if label_col and context_col:
            # Stratified sample
            labels = train_df[label_col].unique()
            per_label = max(1, num_examples // len(labels))
            for label in labels:
                subset = train_df[train_df[label_col] == label]
                sampled = subset.sample(
                    n=min(per_label, len(subset)),
                    random_state=42,
                )
                for _, row in sampled.iterrows():
                    example = {
                        "context": str(row[context_col])[:2000],  # Truncate
                        "target": str(row[query_col]) if query_col else "",
                        "label": str(row[label_col]),
                        "reasoning": f"The context suggests the annotation is '{row[label_col]}' "
                                     f"because the key indicators match this category.",
                    }
                    examples.append(example)

        random.shuffle(examples)
        return examples[:num_examples]


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("=" * 60)
    print("FlagOS Track 3: ICL Automatic Data Annotation")
    print("=" * 60)

    config = Config()

    # Step 1: Discover and load data
    print("\n[Step 1] Loading data...")
    loader = DataLoader(config)
    loader.discover_data_files()

    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    sample_sub = loader.load_sample_submission()

    label_schema = loader.extract_label_schema(train_df)
    icl_examples = loader.prepare_icl_examples(train_df, config.NUM_SHOT_EXAMPLES)

    print(f"\nLabel schema: {label_schema}")
    print(f"ICL examples: {len(icl_examples)}")
    print(f"Test samples: {len(test_df)}")

    # Step 2: Load model
    print("\n[Step 2] Loading model...")
    engine = ICLEngine(config)
    engine.load_model()

    # Step 3: Generate predictions
    print("\n[Step 3] Generating predictions...")
    predictions = []

    # Determine ID column
    id_col = None
    for col in ["ID", "id", "Id", "sample_id", "index"]:
        if col in test_df.columns:
            id_col = col
            break

    total = len(test_df)
    for idx, row in test_df.iterrows():
        sample_id = row[id_col] if id_col else idx

        print(f"\nProcessing {idx + 1}/{total} (ID: {sample_id})...")
        start_time = time.time()

        try:
            result = engine.process_long_context_sample(
                row.to_dict(),
                label_schema=label_schema,
                examples=icl_examples,
            )
            pred_label = result["label"]
            confidence = result["confidence"]
        except Exception as e:
            print(f"  Error: {e}. Using fallback.")
            pred_label = label_schema[0]  # Fallback
            confidence = 0.0

        elapsed = time.time() - start_time
        print(f"  Prediction: {pred_label} (confidence: {confidence:.2f}, time: {elapsed:.1f}s)")

        predictions.append({
            "ID": sample_id,
            "Predicted": pred_label,
        })

        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Step 4: Save submission
    print("\n[Step 4] Saving submission...")
    sub_df = pd.DataFrame(predictions)

    # Ensure correct format
    if not sample_sub.empty:
        # Use sample submission IDs as ground truth
        if "ID" in sample_sub.columns:
            sub_df = sub_df.set_index("ID").reindex(sample_sub["ID"]).reset_index()

    submission_path = os.path.join(config.OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")
    print(f"Total predictions: {len(sub_df)}")
    print(f"\nSubmission preview:")
    print(sub_df.head(10))
    print(f"\nLabel distribution:")
    print(sub_df["Predicted"].value_counts())

    return sub_df


if __name__ == "__main__":
    main()
