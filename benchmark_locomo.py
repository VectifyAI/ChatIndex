"""
ChatIndex LoCoMo Benchmark Script

A simplified benchmarking framework for evaluating ChatIndex on the LoCoMo benchmark.
This script:
1. Builds ChatIndex trees from LoCoMo conversation sessions
2. Answers questions using ChatIndex's query_ctree function
3. Evaluates answers using the evaluation framework from memU-experiment
"""

import json
import os
import sys
import ast
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
dotenv.load_dotenv()

# Add ChatIndex to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ctree import CTree
from retrieval.llm_tools import query_ctree

# Add memU-experiment to path for evaluation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'memU-experiment'))
from evaluate_agent import EvaluateAgent


class ChatIndexLoCoMoTester:
    """
    ChatIndex LoCoMo Benchmark Tester
    
    Processes LoCoMo samples by:
    1. Building ChatIndex trees from conversation sessions
    2. Answering questions using ChatIndex query system
    3. Evaluating answers for accuracy
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        chatindex_model: str = "gpt-4o-mini",
        max_children: int = 10,
        max_workers: int = 3,
        category_filter: Optional[List[str]] = None,
        tree_save_dir: Optional[str] = None
    ):
        """
        Initialize ChatIndex LoCoMo Tester
        
        Args:
            openai_api_key: OpenAI API key for building trees (or set OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key for querying trees (or set ANTHROPIC_API_KEY env var)
            chatindex_model: OpenAI model to use for tree building
            max_children: Maximum children per node in ChatIndex tree
            max_workers: Number of parallel workers for QA processing
            category_filter: Optional list of question categories to test
            tree_save_dir: Optional directory to save built trees
        """
        # Get API keys from args or environment
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")
        
        self.chatindex_model = chatindex_model
        self.max_children = max_children
        self.max_workers = max_workers
        self.category_filter = category_filter
        self.tree_save_dir = Path(tree_save_dir) if tree_save_dir else None
        
        if self.tree_save_dir:
            self.tree_save_dir.mkdir(exist_ok=True)
        
        # Initialize evaluation agent (reuse from memU-experiment)
        self.evaluate_agent = EvaluateAgent(
            azure_endpoint=None,
            api_key=self.openai_api_key,  # EvaluateAgent can use OpenAI
            chat_deployment="gpt-4o",  # Use stronger model for evaluation
            use_entra_id=False,
            api_version="2024-02-15-preview"
        )
        
        self.results = []
        self.processing_time = 0.0
        
        # Initialize error log
        self.log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.error_log_file = f"chatindex_error_log_{self.log_timestamp}.txt"
        self._init_error_log()
        
        print(f"ChatIndex LoCoMo Tester initialized")
        print(f"  - Tree model: {chatindex_model}")
        print(f"  - Max children: {max_children}")
        print(f"  - Max workers: {max_workers}")
        if category_filter:
            print(f"  - Category filter: {category_filter}")
        print(f"  - Error log: {self.error_log_file}")
    
    def _init_error_log(self):
        """Initialize error log file"""
        try:
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                f.write(f"ChatIndex LoCoMo Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Failed to initialize error log: {e}")
    
    def _log_error(self, qa_index: int, question: str, generated_answer: str, 
                   standard_answer: str, category: str, explanation: str = ""):
        """Log error details"""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"QA INDEX: {qa_index + 1}\n")
                f.write(f"CATEGORY: {category}\n")
                f.write(f"QUESTION: {question}\n")
                f.write(f"GENERATED ANSWER: {generated_answer}\n")
                f.write(f"STANDARD ANSWER: {standard_answer}\n")
                f.write(f"EXPLANATION: {explanation}\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            print(f"Failed to write error log: {e}")
    
    def _convert_locomo_to_chatindex_messages(self, conversation_data: Dict) -> List[Dict]:
        """
        Convert LoCoMo conversation format to ChatIndex message format.
        
        LoCoMo format: sessions with utterances containing speaker and text
        ChatIndex format: list of message dicts with role and content
        
        LoCoMo utterances alternate between speaker_a (user) and speaker_b (assistant)
        """
        messages = []
        
        # Extract speaker names
        speaker_a = conversation_data.get('speaker_a', 'Speaker A')
        speaker_b = conversation_data.get('speaker_b', 'Speaker B')
        
        # Find all sessions and sort
        session_keys = [key for key in conversation_data.keys() 
                       if key.startswith('session_') and not key.endswith('_date_time')]
        session_keys.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
        # Process each session
        for session_key in session_keys:
            session_data = conversation_data.get(session_key, [])
            if not session_data:
                continue
            
            # Convert utterances to messages
            # LoCoMo conversations alternate: speaker_a (user) -> speaker_b (assistant) -> speaker_a -> ...
            current_user_msg = None
            
            for utterance in session_data:
                if not isinstance(utterance, dict):
                    continue
                
                speaker = utterance.get('speaker', '')
                text = utterance.get('text', '')
                
                # Skip if no text (might have only image)
                if not text:
                    continue
                
                # Determine role based on speaker
                if speaker == speaker_a:
                    # User message - start a new exchange
                    if current_user_msg:
                        # Previous exchange incomplete, add user message alone
                        messages.append(current_user_msg)
                    current_user_msg = {
                        'role': 'user',
                        'content': text
                    }
                elif speaker == speaker_b:
                    # Assistant message - complete the exchange
                    if current_user_msg:
                        # Complete exchange: user -> assistant
                        messages.append(current_user_msg)
                        messages.append({
                            'role': 'assistant',
                            'content': text
                        })
                        current_user_msg = None
                    else:
                        # No preceding user message, add assistant alone
                        messages.append({
                            'role': 'assistant',
                            'content': text
                        })
            
            # Handle any remaining user message
            if current_user_msg:
                messages.append(current_user_msg)
        
        return messages
    
    def _build_chatindex_tree(self, conversation_data: Dict, sample_id: int) -> Optional[CTree]:
        """
        Build a ChatIndex tree from LoCoMo conversation data.
        
        Returns:
            CTree instance or None if building failed
        """
        try:
            # Convert LoCoMo format to ChatIndex format
            messages = self._convert_locomo_to_chatindex_messages(conversation_data)
            
            if not messages:
                print(f"  Warning: No messages found for sample {sample_id}")
                return None
            
            # Initialize tree
            tree = CTree(
                max_children=self.max_children,
                api_key=self.openai_api_key,
                model=self.chatindex_model
            )
            
            # Build tree incrementally
            # ChatIndex expects messages in groups: [system?, user, assistant]
            # LoCoMo messages alternate: user, assistant, user, assistant, ...
            i = 0
            while i < len(messages):
                group = []
                
                # Check for system message at start (uncommon in LoCoMo, but handle it)
                if i < len(messages) and messages[i]['role'] == 'system':
                    group.append(messages[i])
                    i += 1
                
                # Look for user-assistant pair
                if i < len(messages) and messages[i]['role'] == 'user':
                    group.append(messages[i])
                    i += 1
                    
                    # Add corresponding assistant message if available
                    if i < len(messages) and messages[i]['role'] == 'assistant':
                        group.append(messages[i])
                        i += 1
                        
                        # We have a complete exchange (user + assistant), add to tree
                        try:
                            tree.add(group)
                        except Exception as e:
                            print(f"    Warning: Failed to add message group: {e}")
                            # Continue with next group
                    else:
                        # Missing assistant message, skip this user message
                        # (ChatIndex requires both user and assistant)
                        if group:
                            group.pop()  # Remove the user message we added
                        i += 1
                else:
                    # Unexpected message format, skip
                    i += 1
            
            # Save tree if directory specified
            if self.tree_save_dir:
                save_path = self.tree_save_dir / f"sample_{sample_id}_tree.json"
                tree.save(str(save_path))
            
            return tree
            
        except Exception as e:
            print(f"  Error building tree for sample {sample_id}: {e}")
            return None
    
    def _process_single_qa(self, qa_data: Tuple[str, str, str, int], tree: CTree) -> Dict:
        """
        Process a single QA question using ChatIndex.
        
        Args:
            qa_data: Tuple of (question, answer, category, qa_index)
            tree: CTree instance to query
        
        Returns:
            Result dictionary with answer and evaluation
        """
        question, answer, category, qa_index = qa_data
        
        try:
            # Query the tree using ChatIndex
            result = query_ctree(
                api_key=self.anthropic_api_key,
                ctree=tree,
                user_query=question,
                max_turns=50
            )
            
            if result.get("success"):
                generated_answer = result.get("final_response", "No answer generated")
                turns_used = result.get("turns_used", 0)
            else:
                error_msg = result.get("error", "Unknown error")
                generated_answer = f"Error: {error_msg}"
                turns_used = 0
            
            # Evaluate the answer
            evaluation = self._evaluate_answer(question, generated_answer, answer)
            
            result_dict = {
                'qa_index': qa_index,
                'question': question,
                'generated_answer': generated_answer,
                'standard_answer': answer,
                'category': category,
                'is_correct': evaluation['is_correct'],
                'explanation': evaluation['explanation'],
                'turns_used': turns_used
            }
            
            # Log errors
            if not evaluation['is_correct']:
                self._log_error(
                    qa_index=qa_index,
                    question=question,
                    generated_answer=generated_answer,
                    standard_answer=answer,
                    category=category,
                    explanation=evaluation['explanation']
                )
            
            status = "✓" if evaluation['is_correct'] else "✗"
            print(f"  [{qa_index+1}] {status} Category {category} - {question[:60]}...")
            
            return result_dict
            
        except Exception as e:
            print(f"  Error processing QA {qa_index+1}: {e}")
            return {
                'qa_index': qa_index,
                'question': question,
                'generated_answer': f"Error: {e}",
                'standard_answer': answer,
                'category': category,
                'is_correct': False,
                'explanation': f"Processing failed: {e}",
                'turns_used': 0
            }
    
    def _evaluate_answer(self, question: str, generated_answer: str, standard_answer: str) -> Dict:
        """Evaluate answer using EvaluateAgent"""
        try:
            result = self.evaluate_agent.evaluate_answer_accuracy(question, generated_answer, standard_answer)
            
            if result["success"]:
                return {
                    'is_correct': result['is_correct'],
                    'explanation': result['explanation'],
                    'evaluation_text': result.get('evaluation_text', '')
                }
            else:
                return {
                    'is_correct': False,
                    'explanation': f"Evaluation failed: {result.get('error', 'Unknown error')}",
                    'evaluation_text': ""
                }
        except Exception as e:
            return {
                'is_correct': False,
                'explanation': f"Evaluation failed: {e}",
                'evaluation_text': ""
            }
    
    def _process_qa_parallel(self, qa_data: List[Dict], tree: CTree, max_workers: int = 3) -> List[Dict]:
        """Process multiple QA questions in parallel"""
        if not qa_data:
            return []
        
        # Prepare QA items
        qa_items = []
        for i, qa_item in enumerate(qa_data):
            if 'question' in qa_item and 'answer' in qa_item:
                # Check category filter
                if self.category_filter:
                    item_category = str(qa_item.get('category', 'Unknown'))
                    if item_category not in self.category_filter:
                        continue
                
                qa_items.append((
                    qa_item['question'],
                    qa_item['answer'],
                    qa_item.get('category', 'Unknown'),
                    i
                ))
        
        if not qa_items:
            return []
        
        question_results = []
        completed_count = 0
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_qa = {
                executor.submit(self._process_single_qa, qa_item, tree): qa_item
                for qa_item in qa_items
            }
            
            for future in as_completed(future_to_qa):
                completed_count += 1
                try:
                    result = future.result()
                    question_results.append(result)
                except Exception as e:
                    qa_item = future_to_qa[future]
                    print(f"  Exception processing QA {qa_item[3]+1}: {e}")
                    question_results.append({
                        'qa_index': qa_item[3],
                        'question': qa_item[0],
                        'generated_answer': f"Error: {e}",
                        'standard_answer': qa_item[1],
                        'category': qa_item[2],
                        'is_correct': False,
                        'explanation': f"Exception: {e}",
                        'turns_used': 0
                    })
        
        # Sort by qa_index
        question_results.sort(key=lambda x: x['qa_index'])
        
        successful_qa = sum(1 for r in question_results if r['is_correct'])
        print(f"  Completed {completed_count} questions: {successful_qa}/{len(question_results)} correct")
        
        return question_results
    
    def process_sample(self, sample: Dict, sample_id: int) -> Dict:
        """Process one LoCoMo sample"""
        start_time = time.time()
        
        try:
            conversation_data = sample['conversation']
            qa_data = sample.get('qa', [])
            
            print(f"\n=== Processing Sample {sample_id} ===")
            
            # Build ChatIndex tree
            print(f"  Building ChatIndex tree...")
            tree = self._build_chatindex_tree(conversation_data, sample_id)
            
            if not tree:
                return {
                    'sample_id': sample_id,
                    'success': False,
                    'error': 'Failed to build tree',
                    'question_results': [],
                    'category_stats': {},
                    'processing_time': time.time() - start_time
                }
            
            print(f"  Tree built successfully")
            
            # Process QA questions
            print(f"  Processing {len(qa_data)} questions...")
            question_results = self._process_qa_parallel(qa_data, tree, self.max_workers)
            
            # Calculate category statistics
            category_stats = {}
            for result in question_results:
                category = result['category']
                if category not in category_stats:
                    category_stats[category] = {'total': 0, 'correct': 0}
                category_stats[category]['total'] += 1
                if result['is_correct']:
                    category_stats[category]['correct'] += 1
            
            processing_time = time.time() - start_time
            
            return {
                'sample_id': sample_id,
                'success': True,
                'questions_total': len(qa_data),
                'questions_processed': len(question_results),
                'question_results': question_results,
                'category_stats': category_stats,
                'processing_time': processing_time
            }
            
        except Exception as e:
            print(f"  Error processing sample {sample_id}: {e}")
            return {
                'sample_id': sample_id,
                'success': False,
                'error': str(e),
                'question_results': [],
                'category_stats': {},
                'processing_time': time.time() - start_time
            }
    
    def run_test(self, data_file: str, sample_use: Optional[str] = None) -> Dict:
        """Run the benchmark test"""
        start_time = time.time()
        
        try:
            # Load data
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter samples if specified
            if sample_use:
                try:
                    parsed_value = ast.literal_eval(sample_use)
                    
                    if isinstance(parsed_value, int):
                        data = data[:parsed_value]
                        print(f"Using first {parsed_value} samples")
                    elif isinstance(parsed_value, list):
                        valid_indices = [i for i in parsed_value if isinstance(i, int) and 0 <= i < len(data)]
                        data = [data[i] for i in valid_indices]
                        print(f"Using samples at indices: {valid_indices}")
                    else:
                        raise ValueError("sample_use must be integer or list")
                except Exception as e:
                    print(f"Error parsing sample_use: {e}, using all samples")
            else:
                print(f"Using all {len(data)} samples")
            
            # Process each sample
            all_results = []
            overall_category_stats = {}
            total_questions = 0
            total_correct = 0
            
            for i, sample in enumerate(data, 1):
                result = self.process_sample(sample, i)
                all_results.append(result)
                
                if result['success']:
                    # Aggregate statistics
                    for category, stats in result['category_stats'].items():
                        if category not in overall_category_stats:
                            overall_category_stats[category] = {'total': 0, 'correct': 0}
                        overall_category_stats[category]['total'] += stats['total']
                        overall_category_stats[category]['correct'] += stats['correct']
                    
                    total_questions += result['questions_processed']
                    total_correct += sum(1 for qr in result['question_results'] if qr['is_correct'])
                
                print(f"Sample {i} completed in {result['processing_time']:.2f}s")
            
            total_time = time.time() - start_time
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
            
            # Calculate category accuracies
            category_accuracies = {}
            for category, stats in overall_category_stats.items():
                accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
                category_accuracies[category] = accuracy
            
            summary = {
                'total_samples': len(data),
                'successful_samples': sum(1 for r in all_results if r['success']),
                'total_questions': total_questions,
                'total_correct': total_correct,
                'overall_accuracy': overall_accuracy,
                'category_stats': overall_category_stats,
                'category_accuracies': category_accuracies,
                'total_time': total_time,
                'avg_time_per_sample': total_time / len(data) if data else 0.0
            }
            
            self.results = all_results
            self.processing_time = total_time
            
            return {
                'success': True,
                'summary': summary,
                'detailed_results': all_results
            }
            
        except Exception as e:
            print(f"Test run failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': {},
                'detailed_results': []
            }
    
    def print_results(self):
        """Print test results"""
        if not self.results:
            print("No results to display")
            return
        
        summary = {
            'total_samples': len(self.results),
            'successful_samples': sum(1 for r in self.results if r['success']),
            'total_questions': sum(r.get('questions_processed', 0) for r in self.results if r['success']),
            'total_correct': sum(sum(1 for qr in r.get('question_results', []) if qr['is_correct']) 
                                for r in self.results if r['success']),
        }
        
        total_questions = summary['total_questions']
        total_correct = summary['total_correct']
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Aggregate category stats
        category_stats = {}
        for result in self.results:
            if result['success']:
                for category, stats in result['category_stats'].items():
                    if category not in category_stats:
                        category_stats[category] = {'total': 0, 'correct': 0}
                    category_stats[category]['total'] += stats['total']
                    category_stats[category]['correct'] += stats['correct']
        
        print(f"\n{'='*60}")
        print(f"CHATINDEX LOCOMO BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Samples processed: {summary['successful_samples']}/{summary['total_samples']}")
        print(f"Total questions: {total_questions}")
        print(f"Total correct: {total_correct}")
        print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"Total time: {self.processing_time:.2f}s")
        
        print(f"\n{'='*60}")
        print(f"CATEGORY-WISE ACCURACY")
        print(f"{'='*60}")
        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            print(f"Category {category}: {stats['correct']:3}/{stats['total']:3} ({accuracy:.1%})")
        
        print(f"\n{'='*60}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChatIndex LoCoMo Benchmark')
    parser.add_argument('--data-file', default='../memU-experiment/data/locomo10.json',
                       help='Path to LoCoMo test data file')
    parser.add_argument('--sample-use', type=str,
                       help='Sample indices: number (e.g., "5") or list (e.g., "[0,1,3]")')
    parser.add_argument('--chatindex-model', default='gpt-4o-mini',
                       help='OpenAI model for tree building')
    parser.add_argument('--max-children', type=int, default=10,
                       help='Max children per node in ChatIndex tree')
    parser.add_argument('--max-workers', type=int, default=3,
                       help='Number of parallel workers for QA processing')
    parser.add_argument('--category', type=str,
                       help='Filter by category (e.g., "1" or "0,2,3")')
    parser.add_argument('--tree-save-dir', type=str,
                       help='Directory to save built trees')
    
    args = parser.parse_args()
    
    # Parse category filter
    category_filter = None
    if args.category:
        if ',' in args.category:
            category_filter = [cat.strip() for cat in args.category.split(',')]
        else:
            category_filter = [args.category.strip()]
    
    # Initialize tester
    tester = ChatIndexLoCoMoTester(
        chatindex_model=args.chatindex_model,
        max_children=args.max_children,
        max_workers=args.max_workers,
        category_filter=category_filter,
        tree_save_dir=args.tree_save_dir
    )
    
    # Run test
    results = tester.run_test(args.data_file, args.sample_use)
    
    if results['success']:
        tester.print_results()
        
        # Save results
        output_file = f"chatindex_locomo_results_{tester.log_timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Error log: {tester.error_log_file}")
    else:
        print(f"Test failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

