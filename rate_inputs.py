#!/usr/bin/env python3
"""
Rate input files for training quality.
Checks DSL format, diversity, quality, and provides recommendations.
"""

import os
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from dsl import DSLDecoder

class InputFileRater:
    """Rate input files for training quality"""
    
    def __init__(self):
        self.dsl_decoder = DSLDecoder()
        self.issues = []
        self.stats = {}
    
    def rate_file(self, filepath: str) -> Dict:
        """Rate a single input file"""
        if not os.path.exists(filepath):
            return {'error': f'File not found: {filepath}'}
        
        self.issues = []
        self.stats = {
            'filename': os.path.basename(filepath),
            'total_lines': 0,
            'question_answer_pairs': 0,
            'valid_dsl': 0,
            'invalid_dsl': 0,
            'duplicate_pairs': 0,
            'response_types': Counter(),
            'tool_types': Counter(),
            'avg_question_length': 0,
            'avg_response_length': 0,
            'speech_compatible': 0,
            'speech_issues': 0,
            'format_issues': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        pairs = self._parse_file(filepath)
        self.stats['question_answer_pairs'] = len(pairs)
        
        if len(pairs) == 0:
            self.stats['quality_score'] = 0.0
            self.stats['recommendations'].append('File has no valid question-answer pairs')
            return self.stats
        
        # Analyze pairs
        self._analyze_dsl_validity(pairs)
        self._analyze_duplicates(pairs)
        self._analyze_diversity(pairs)
        self._analyze_speech_compatibility(pairs)
        self._analyze_format_issues(pairs)
        self._calculate_quality_score()
        
        return self.stats
    
    def _parse_file(self, filepath: str) -> List[Tuple[str, str]]:
        """Parse file into question-answer pairs"""
        pairs = []
        current_question = None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total_lines'] += 1
                line = line.strip()
                
                if not line:
                    if current_question:
                        current_question = None
                    continue
                
                # Check if it's a DSL response
                if line.startswith(('S:', 'T:', 'C:', 'E:', 'CL')) or line == 'CL':
                    if current_question:
                        pairs.append((current_question, line))
                        current_question = None
                # Check for CTX: format (context file)
                elif line.startswith('CTX:') or line.startswith('CXT:'):
                    # Context line - next line should be response
                    current_question = line
                # Check for |R: format (multi-tool file)
                elif '|R:' in line or '|S:' in line:
                    # This is a question with results - next line should be response
                    current_question = line
                else:
                    # Regular question line
                    current_question = line
        
        return pairs
    
    def _analyze_dsl_validity(self, pairs: List[Tuple[str, str]]):
        """Check if DSL responses are valid"""
        for question, answer in pairs:
            # Try to decode DSL
            decoded = self.dsl_decoder.decode(answer)
            if decoded:
                self.stats['valid_dsl'] += 1
                # Track response types
                if decoded.get('type') == 'response':
                    self.stats['response_types']['S'] += 1
                elif decoded.get('type') == 'tool':
                    self.stats['response_types']['T'] += 1
                    tool_name = decoded.get('tool', '')
                    if tool_name:
                        self.stats['tool_types'][tool_name] += 1
                elif decoded.get('type') == 'command':
                    self.stats['response_types']['C'] += 1
                elif decoded.get('type') == 'cloud':
                    self.stats['response_types']['CL'] += 1
                elif decoded.get('type') == 'error':
                    self.stats['response_types']['E'] += 1
                elif decoded.get('type') == 'tool_chain':
                    self.stats['response_types']['T_CHAIN'] += 1
                elif decoded.get('type') == 'response_command':
                    self.stats['response_types']['S+C'] += 1
            else:
                self.stats['invalid_dsl'] += 1
                self.stats['format_issues'].append(f'Invalid DSL: {answer[:50]}')
    
    def _analyze_duplicates(self, pairs: List[Tuple[str, str]]):
        """Find duplicate question-answer pairs"""
        seen = set()
        duplicates = []
        
        for question, answer in pairs:
            key = (question.lower().strip(), answer.strip())
            if key in seen:
                duplicates.append((question, answer))
            else:
                seen.add(key)
        
        self.stats['duplicate_pairs'] = len(duplicates)
        if len(duplicates) > len(pairs) * 0.1:  # More than 10% duplicates
            self.stats['recommendations'].append(
                f'High duplicate rate: {len(duplicates)}/{len(pairs)} ({100*len(duplicates)/len(pairs):.1f}%)'
            )
    
    def _analyze_diversity(self, pairs: List[Tuple[str, str]]):
        """Analyze diversity of questions and responses"""
        question_lengths = []
        response_lengths = []
        unique_questions = set()
        
        for question, answer in pairs:
            question_lengths.append(len(question))
            response_lengths.append(len(answer))
            unique_questions.add(question.lower().strip())
        
        self.stats['avg_question_length'] = sum(question_lengths) / len(question_lengths) if question_lengths else 0
        self.stats['avg_response_length'] = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        self.stats['unique_questions'] = len(unique_questions)
        self.stats['diversity_ratio'] = len(unique_questions) / len(pairs) if pairs else 0
        
        if self.stats['diversity_ratio'] < 0.7:
            self.stats['recommendations'].append(
                f'Low diversity: only {self.stats["diversity_ratio"]*100:.1f}% unique questions'
            )
    
    def _analyze_speech_compatibility(self, pairs: List[Tuple[str, str]]):
        """Check if text is compatible with speech-to-text/text-to-speech"""
        # Characters that cause issues with speech
        problematic_chars = re.compile(r'[!?\'"`]')
        
        for question, answer in pairs:
            # Check question
            if not problematic_chars.search(question):
                self.stats['speech_compatible'] += 1
            else:
                self.stats['speech_issues'] += 1
            
            # Check if DSL markers are uppercase but content is lowercase
            # Extract content after DSL marker
            if ':' in answer:
                marker, content = answer.split(':', 1)
                # Check if content has uppercase (except DSL markers)
                if content and content != content.lower():
                    # Check if it's just DSL markers or actual content
                    if not re.match(r'^[A-Z]+:', content):
                        self.stats['speech_issues'] += 1
        
        speech_ratio = self.stats['speech_compatible'] / len(pairs) if pairs else 0
        if speech_ratio < 0.9:
            self.stats['recommendations'].append(
                f'Speech compatibility issues: {self.stats["speech_issues"]} issues found'
            )
    
    def _analyze_format_issues(self, pairs: List[Tuple[str, str]]):
        """Find format and quality issues"""
        # Check for common issues
        for question, answer in pairs:
            # Check for suspicious patterns
            if 'okcmove' in answer.lower():
                self.stats['format_issues'].append('Found "okcmove" - likely typo, should be "S:ok;C:move"')
            
            # Check for missing spaces
            if re.search(r'[a-z][A-Z]', answer):
                # Lowercase followed by uppercase (might be missing space)
                if not answer.startswith(('S:', 'T:', 'C:', 'E:')):
                    self.stats['format_issues'].append(f'Possible formatting issue: {answer[:50]}')
            
            # Check for very short questions
            if len(question.strip()) < 3:
                self.stats['format_issues'].append(f'Very short question: "{question}"')
            
            # Check for very long questions
            if len(question) > 200:
                self.stats['format_issues'].append(f'Very long question ({len(question)} chars): {question[:50]}...')
    
    def _calculate_quality_score(self):
        """Calculate overall quality score (0-100)"""
        score = 100.0
        total_pairs = self.stats['question_answer_pairs']
        
        if total_pairs == 0:
            self.stats['quality_score'] = 0.0
            return
        
        # Deduct points for issues
        # Invalid DSL (major issue)
        if self.stats['invalid_dsl'] > 0:
            invalid_ratio = self.stats['invalid_dsl'] / total_pairs
            score -= invalid_ratio * 30  # Up to 30 points
        
        # Duplicates (moderate issue)
        if self.stats['duplicate_pairs'] > 0:
            dup_ratio = self.stats['duplicate_pairs'] / total_pairs
            if dup_ratio > 0.1:
                score -= min(dup_ratio * 20, 20)  # Up to 20 points
        
        # Low diversity (moderate issue)
        if self.stats.get('diversity_ratio', 1.0) < 0.7:
            score -= (0.7 - self.stats['diversity_ratio']) * 15  # Up to 15 points
        
        # Speech compatibility (minor issue)
        if self.stats['speech_issues'] > 0:
            speech_ratio = self.stats['speech_issues'] / total_pairs
            score -= speech_ratio * 10  # Up to 10 points
        
        # Format issues (minor issue)
        if len(self.stats['format_issues']) > 0:
            score -= min(len(self.stats['format_issues']) * 2, 15)  # Up to 15 points
        
        # Bonus for good diversity
        if self.stats.get('diversity_ratio', 0) > 0.9:
            score += 5
        
        # Ensure score is between 0 and 100
        self.stats['quality_score'] = max(0.0, min(100.0, score))
        
        # Add rating label
        if self.stats['quality_score'] >= 90:
            self.stats['rating'] = 'Excellent'
        elif self.stats['quality_score'] >= 75:
            self.stats['rating'] = 'Good'
        elif self.stats['quality_score'] >= 60:
            self.stats['rating'] = 'Fair'
        elif self.stats['quality_score'] >= 40:
            self.stats['rating'] = 'Poor'
        else:
            self.stats['rating'] = 'Very Poor'
    
    def print_rating(self, stats: Dict):
        """Print formatted rating report"""
        print(f"\n{'='*70}")
        print(f"File: {stats['filename']}")
        print(f"{'='*70}")
        print(f"Quality Score: {stats['quality_score']:.1f}/100 - {stats.get('rating', 'N/A')}")
        print()
        
        print("Statistics:")
        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Question-Answer pairs: {stats['question_answer_pairs']}")
        print(f"  Unique questions: {stats.get('unique_questions', 0)}")
        print(f"  Diversity ratio: {stats.get('diversity_ratio', 0)*100:.1f}%")
        print()
        
        print("DSL Validity:")
        print(f"  Valid DSL: {stats['valid_dsl']}")
        print(f"  Invalid DSL: {stats['invalid_dsl']}")
        if stats['invalid_dsl'] > 0:
            print(f"  ⚠️  Warning: {stats['invalid_dsl']} invalid DSL entries found")
        print()
        
        print("Response Types:")
        for rtype, count in stats['response_types'].most_common():
            pct = (count / stats['question_answer_pairs'] * 100) if stats['question_answer_pairs'] > 0 else 0
            print(f"  {rtype}: {count} ({pct:.1f}%)")
        print()
        
        if stats['tool_types']:
            print("Tool Types (top 10):")
            for tool, count in stats['tool_types'].most_common(10):
                print(f"  {tool}: {count}")
            print()
        
        print("Quality Metrics:")
        print(f"  Duplicate pairs: {stats['duplicate_pairs']}")
        print(f"  Speech compatible: {stats['speech_compatible']}/{stats['question_answer_pairs']}")
        print(f"  Speech issues: {stats['speech_issues']}")
        print(f"  Avg question length: {stats['avg_question_length']:.1f} chars")
        print(f"  Avg response length: {stats['avg_response_length']:.1f} chars")
        print()
        
        if stats['format_issues']:
            print("Format Issues:")
            for issue in stats['format_issues'][:10]:  # Show first 10
                print(f"  ⚠️  {issue}")
            if len(stats['format_issues']) > 10:
                print(f"  ... and {len(stats['format_issues']) - 10} more issues")
            print()
        
        if stats['recommendations']:
            print("Recommendations:")
            for rec in stats['recommendations']:
                print(f"  💡 {rec}")
            print()


def rate_all_files(inputs_dir='inputs', verbose=True) -> Dict:
    """Rate all input files and return summary"""
    rater = InputFileRater()
    
    # Get all input files
    if not os.path.exists(inputs_dir):
        if verbose:
            print(f"Warning: {inputs_dir} directory not found")
        return {'error': f'Directory not found: {inputs_dir}'}
    
    input_files = []
    for filename in os.listdir(inputs_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(inputs_dir, filename)
            input_files.append(filepath)
    
    if not input_files:
        if verbose:
            print(f"Warning: No .txt files found in {inputs_dir}")
        return {'error': 'No input files found'}
    
    input_files.sort()
    
    if verbose:
        print("="*70)
        print("INPUT FILES QUALITY RATING REPORT")
        print("="*70)
    
    all_stats = []
    for filepath in input_files:
        stats = rater.rate_file(filepath)
        if 'error' not in stats:
            all_stats.append(stats)
            if verbose:
                rater.print_rating(stats)
    
    # Summary
    if verbose and all_stats:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Filename':<35} {'Pairs':<8} {'Score':<8} {'Rating':<12}")
        print("-"*70)
        
        for stats in sorted(all_stats, key=lambda x: x['quality_score'], reverse=True):
            filename = stats['filename'][:34]
            pairs = stats['question_answer_pairs']
            score = stats['quality_score']
            rating = stats.get('rating', 'N/A')
            print(f"{filename:<35} {pairs:<8} {score:<8.1f} {rating:<12}")
        
        print("\n" + "="*70)
        total_pairs = sum(s['question_answer_pairs'] for s in all_stats)
        avg_score = sum(s['quality_score'] for s in all_stats) / len(all_stats) if all_stats else 0
        print(f"Total pairs across all files: {total_pairs}")
        print(f"Average quality score: {avg_score:.1f}/100")
        print("="*70)
    
    # Return summary for programmatic use
    return {
        'files': all_stats,
        'total_pairs': sum(s['question_answer_pairs'] for s in all_stats),
        'avg_score': sum(s['quality_score'] for s in all_stats) / len(all_stats) if all_stats else 0,
        'min_score': min((s['quality_score'] for s in all_stats), default=0),
        'max_score': max((s['quality_score'] for s in all_stats), default=0),
    }


def main():
    """Main entry point"""
    rate_all_files(verbose=True)


if __name__ == '__main__':
    main()

