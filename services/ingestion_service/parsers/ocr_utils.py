# base/parsers/ocr_utils.py
"""
Utility functions for OCR text processing.
Contains common text cleaning utilities used by OCR parsers.
"""
import re


def clean_ocr_repetitions(text: str) -> str:
    """
    Remove repetitive patterns from OCR text.
    
    OCR can sometimes produce repeated phrases/words due to scanning artifacts.
    This function detects and removes such repetitions.
    
    Args:
        text: The OCR-extracted text to clean
        
    Returns:
        Cleaned text with repetitions removed
    """
    if not text:
        return text
        
    # Clean up common OCR repetition patterns
    # Look for repeated phrases of 5-15 words
    words = text.split()
    cleaned_words = []
    i = 0
    
    while i < len(words):
        cleaned_words.append(words[i])
        
        # Check for repetitions of phrases (5-15 words long)
        for phrase_len in range(5, min(16, len(words) - i)):
            phrase = words[i:i+phrase_len]
            phrase_str = ' '.join(phrase)
            
            # Look ahead to see if this phrase repeats immediately
            next_pos = i + phrase_len
            repeat_count = 0
            
            while next_pos + phrase_len <= len(words):
                next_phrase = words[next_pos:next_pos+phrase_len]
                next_phrase_str = ' '.join(next_phrase)
                
                if next_phrase_str == phrase_str:
                    repeat_count += 1
                    next_pos += phrase_len
                else:
                    break
            
            # If we found repetitions, skip them in the output
            if repeat_count > 0:
                print(f"Found {repeat_count} repetitions of phrase: '{phrase_str}'")
                i = next_pos - 1  # -1 because we'll increment i at the end of the loop
                break
        
        i += 1
    
    return ' '.join(cleaned_words)
