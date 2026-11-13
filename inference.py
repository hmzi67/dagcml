"""
Complete Inference Script for Curriculum Meta-Learning NLI Model
FIXED: Handles Config object serialization issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.deberta_v2 import DebertaV2Tokenizer
import numpy as np
from typing import List, Dict, Union
import os

# ============================================================================
# CONFIG CLASS (must match training config)
# ============================================================================

class Config:
    """Configuration matching paper specifications exactly"""
    model_name = 'microsoft/mdeberta-v3-base'
    num_labels = 3
    max_length = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# MODEL DEFINITION (must match training)
# ============================================================================

class MetaNLIModel(nn.Module):
    """mDeBERTa model for NLI"""
    
    def __init__(self, model_name='microsoft/mdeberta-v3-base', num_labels=3):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# ============================================================================
# INFERENCE CLASS
# ============================================================================

class NLIInference:
    """
    Inference wrapper for trained CML model
    """
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize inference model
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        print(f"Loading model from: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        # Load tokenizer (use slow tokenizer to avoid tiktoken conversion issues)
        try:
            # Try loading slow tokenizer directly
            self.tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/mdeberta-v3-base')
            print("✓ Loaded slow tokenizer")
        except Exception as e:
            print(f"Warning: Could not load slow tokenizer: {e}")
            # Fallback: Try to force slow tokenizer via AutoTokenizer
            try:
                # Force use of slow tokenizer by specifying tokenizer_class
                from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
                # Get the slow tokenizer class
                tokenizer_config = AutoTokenizer.get_tokenizer_config('microsoft/mdeberta-v3-base')
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'microsoft/mdeberta-v3-base', 
                    use_fast=False,
                    local_files_only=False
                )
            except Exception as e2:
                print(f"Error loading tokenizer: {e2}")
                raise
        
        # Load model
        self.model = MetaNLIModel().to(self.device)
        
        # Load checkpoint with proper handling
        try:
            # First try loading with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except:
            # If that fails, load with weights_only=False but only extract state_dict
            print("Loading checkpoint with full unpickling...")
            checkpoint_raw = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Extract only the model weights, ignore Config and other objects
            if isinstance(checkpoint_raw, dict) and 'model_state_dict' in checkpoint_raw:
                checkpoint = {'model_state_dict': checkpoint_raw['model_state_dict']}
                
                # Print additional info if available
                if 'phase' in checkpoint_raw:
                    print(f"  Phase: {checkpoint_raw['phase']}")
                if 'epoch' in checkpoint_raw:
                    print(f"  Epoch: {checkpoint_raw['epoch']}")
                if 'episode' in checkpoint_raw:
                    print(f"  Episode: {checkpoint_raw['episode']}")
            else:
                checkpoint = checkpoint_raw
        
        # Load state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("✓ Model loaded successfully!")
    
    def predict_single(self, premise: str, hypothesis: str, return_probs: bool = False):
        """
        Predict relationship between premise and hypothesis
        
        Args:
            premise: The premise sentence
            hypothesis: The hypothesis sentence
            return_probs: If True, return probability distribution
        
        Returns:
            dict: Prediction results
        """
        # Tokenize
        encoded = self.tokenizer(
            premise,
            hypothesis,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
        
        result = {
            'premise': premise,
            'hypothesis': hypothesis,
            'prediction': self.label_map[pred_label],
            'prediction_id': pred_label
        }
        
        if return_probs:
            result['probabilities'] = {
                'entailment': probs[0][0].item(),
                'neutral': probs[0][1].item(),
                'contradiction': probs[0][2].item()
            }
            result['confidence'] = probs[0][pred_label].item()
        
        return result
    
    def predict_batch(self, premises: List[str], hypotheses: List[str], 
                     batch_size: int = 32, return_probs: bool = False):
        """
        Predict for multiple premise-hypothesis pairs
        
        Args:
            premises: List of premise sentences
            hypotheses: List of hypothesis sentences
            batch_size: Batch size for processing
            return_probs: If True, return probability distributions
        
        Returns:
            list: List of prediction results
        """
        assert len(premises) == len(hypotheses), "Premises and hypotheses must have same length"
        
        results = []
        
        for i in range(0, len(premises), batch_size):
            batch_premises = premises[i:i+batch_size]
            batch_hypotheses = hypotheses[i:i+batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
                pred_labels = torch.argmax(probs, dim=-1)
            
            # Process results
            for j in range(len(batch_premises)):
                result = {
                    'premise': batch_premises[j],
                    'hypothesis': batch_hypotheses[j],
                    'prediction': self.label_map[pred_labels[j].item()],
                    'prediction_id': pred_labels[j].item()
                }
                
                if return_probs:
                    result['probabilities'] = {
                        'entailment': probs[j][0].item(),
                        'neutral': probs[j][1].item(),
                        'contradiction': probs[j][2].item()
                    }
                    result['confidence'] = probs[j][pred_labels[j]].item()
                
                results.append(result)
        
        return results
    
    def evaluate_dataset(self, dataset, batch_size: int = 32):
        """
        Evaluate on a dataset (expects 'premise', 'hypothesis', 'label' keys)
        
        Args:
            dataset: Dataset with premise, hypothesis, and label
            batch_size: Batch size for processing
        
        Returns:
            dict: Evaluation metrics
        """
        all_preds = []
        all_labels = []
        
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset.select(range(i, min(i+batch_size, len(dataset))))
            
            premises = list(batch_data['premise'])
            hypotheses = list(batch_data['hypothesis'])
            labels = list(batch_data['label'])
            
            # Tokenize
            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels)
        
        # Per-class accuracy
        class_accuracies = {}
        for label_id, label_name in self.label_map.items():
            mask = all_labels == label_id
            if mask.sum() > 0:
                class_acc = np.mean(all_preds[mask] == all_labels[mask])
                class_accuracies[label_name] = class_acc
        
        return {
            'accuracy': accuracy,
            'total_samples': len(all_labels),
            'class_accuracies': class_accuracies
        }
    
    def predict_from_file(self, input_file: str, output_file: str = None, batch_size: int = 32):
        """
        Predict from CSV file with premise and hypothesis columns
        
        Args:
            input_file: Path to CSV file with 'premise' and 'hypothesis' columns
            output_file: Path to save results (optional)
            batch_size: Batch size for processing
        
        Returns:
            pandas.DataFrame: Results with predictions
        """
        import pandas as pd
        
        # Read input file
        df = pd.read_csv(input_file)
        
        if 'premise' not in df.columns or 'hypothesis' not in df.columns:
            raise ValueError("CSV must have 'premise' and 'hypothesis' columns")
        
        # Get predictions
        results = self.predict_batch(
            df['premise'].tolist(),
            df['hypothesis'].tolist(),
            batch_size=batch_size,
            return_probs=True
        )
        
        # Add results to dataframe
        df['prediction'] = [r['prediction'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['prob_entailment'] = [r['probabilities']['entailment'] for r in results]
        df['prob_neutral'] = [r['probabilities']['neutral'] for r in results]
        df['prob_contradiction'] = [r['probabilities']['contradiction'] for r in results]
        
        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"✓ Results saved to: {output_file}")
        
        return df

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Examples of how to use the inference class"""
    
    # ========================================================================
    # 1. INITIALIZE MODEL
    # ========================================================================
    
    # Change this to your checkpoint path
    checkpoint_path = 'checkpoints/meta_episode_400.pt'
    
    # Initialize inference
    inferencer = NLIInference(checkpoint_path, device='cuda')
    
    # ========================================================================
    # 2. SINGLE PREDICTION
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Prediction")
    print("="*70)
    
    premise = "A person on a horse jumps over a broken down airplane."
    hypothesis = "A person is training his horse for a competition."
    
    result = inferencer.predict_single(premise, hypothesis, return_probs=True)
    
    print(f"\nPremise: {result['premise']}")
    print(f"Hypothesis: {result['hypothesis']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nProbabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.4f}")
    
    # ========================================================================
    # 3. BATCH PREDICTION
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Prediction")
    print("="*70)
    
    premises = [
        "I love you.",
        "My name is Hamza.",
        "Scientists discovered a new planet."
    ]
    
    hypotheses = [
        "I hate you",
        "Mera name hamza hai.",
        "A new star was found."
    ]
    
    results = inferencer.predict_batch(premises, hypotheses, return_probs=True)
    
    for i, result in enumerate(results):
        print(f"\n[Example {i+1}]")
        print(f"Premise: {result['premise']}")
        print(f"Hypothesis: {result['hypothesis']}")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
    
    # ========================================================================
    # 4. DATASET EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Dataset Evaluation")
    print("="*70)
    
    try:
        from datasets import load_dataset
        
        # Load English test set
        test_dataset = load_dataset('facebook/xnli', 'en', split='test')
        
        # Take a small subset for demo
        test_dataset = test_dataset.select(range(100))
        
        # Evaluate
        metrics = inferencer.evaluate_dataset(test_dataset, batch_size=32)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"\nPer-class Accuracies:")
        for label, acc in metrics['class_accuracies'].items():
            print(f"  {label}: {acc*100:.2f}%")
    
    except Exception as e:
        print(f"Dataset evaluation skipped: {e}")
    
    # ========================================================================
    # 5. CSV FILE PREDICTION (NEW!)
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXAMPLE 4: CSV File Prediction")
    print("="*70)
    
    # Example: Create a sample CSV
    import pandas as pd
    
    sample_data = {
        'premise': [
            "A person is riding a bike.",
            "The sky is blue.",
            "A dog is barking loudly."
        ],
        'hypothesis': [
            "Someone is cycling.",
            "It's nighttime.",
            "A cat is meowing."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_input.csv', index=False)
    
    # Predict from CSV
    results_df = inferencer.predict_from_file('sample_input.csv', 'sample_output.csv')
    
    print("\nResults:")
    print(results_df[['premise', 'hypothesis', 'prediction', 'confidence']].to_string(index=False))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("CML MODEL INFERENCE")
    print("="*70)
    
    # Run examples
    example_usage()
    
    print("\n" + "="*70)
    print("INFERENCE COMPLETE!")
    print("="*70)
    
    