"""
OCR extraction using DeepSeek-OCR integration.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from deepcompress.core.config import DeepCompressConfig
from deepcompress.exceptions import GPUError, OCRError
from deepcompress.models.document import ExtractedDocument, Page


class OCRExtractor:
    """
    DeepSeek-OCR integration for vision-based document extraction.

    Uses a 3B parameter vision-language model with:
    - SAM-base vision encoder
    - CLIP-large global attention
    - MoE decoder (64 experts, 6 active)
    - 16× compression of vision tokens
    """

    def __init__(self, config: DeepCompressConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = config.ocr_device

    async def initialize(self) -> None:
        """
        Initialize the OCR model and tokenizer.

        Loads DeepSeek-OCR model onto GPU with bfloat16 precision.
        """
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            import warnings
            import logging
            import os
            from deepcompress.utils.logging import setup_model_warning_suppression
            
            logger = logging.getLogger(__name__)
            
            # Log initialization
            logger.info(f"Initializing DeepSeek-OCR on device: {self._device}")
            logger.info(f"Model: {self.config.ocr_model}, Mode: {self.config.ocr_mode}")
            
            # Setup comprehensive warning suppression
            if self.config.suppress_model_warnings:
                setup_model_warning_suppression(suppress=True)
                logger.debug("Model warnings suppression enabled")
            
            # Disable tqdm globally for this process
            os.environ['TQDM_DISABLE'] = '1'
            
            # Apply compatibility patch for newer transformers versions
            logger.info("Applying transformers compatibility patches...")
            self._apply_transformers_compatibility_patch()

            # DeepSeek-OCR uses AutoTokenizer, not AutoProcessor
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.ocr_model,
                revision=self.config.ocr_model_revision,
                trust_remote_code=True,
            )
            
            # Configure tokenizer properly to avoid warnings
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                else:
                    # Add a default pad token if none exists
                    self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids('[PAD]')
            
            # Ensure padding_side is set
            if not hasattr(self._tokenizer, 'padding_side') or self._tokenizer.padding_side is None:
                self._tokenizer.padding_side = 'left'
            
            # Log tokenizer configuration
            logger.info(f"Tokenizer configured: pad_token={self._tokenizer.pad_token}, "
                       f"pad_token_id={self._tokenizer.pad_token_id}, "
                       f"eos_token_id={self._tokenizer.eos_token_id}")

            # Determine device_map for optimal loading
            device_map = None
            if self._device.startswith("cuda"):
                # Load directly on specified GPU
                device_map = {"": self._device}
                
                # Set GPU memory fraction if configured
                if self.config.gpu_memory_fraction < 1.0:
                    device_idx = int(self._device.split(":")[-1]) if ":" in self._device else 0
                    torch.cuda.set_per_process_memory_fraction(
                        self.config.gpu_memory_fraction,
                        device=device_idx,
                    )
            
            # Base model kwargs
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.config.use_bfloat16 else torch.float32,
                "trust_remote_code": True,
                "revision": self.config.ocr_model_revision,
                "low_cpu_mem_usage": True,  # Optimize memory usage during loading
            }
            
            # Add device_map if using CUDA
            if device_map is not None:
                model_kwargs["device_map"] = device_map

            # Attempt to use flash attention 2 if available and enabled
            if self.config.enable_flash_attention and self._device.startswith("cuda"):
                try:
                    self._model = AutoModel.from_pretrained(
                        self.config.ocr_model,
                        attn_implementation="flash_attention_2",
                        **model_kwargs,
                    )
                except (ImportError, ValueError, Exception) as e:
                    # Fall back to eager attention if flash attention not available
                    warnings.warn(f"Flash Attention 2 not available, falling back to eager: {e}")
                    model_kwargs.pop("attn_implementation", None)
                    self._model = AutoModel.from_pretrained(
                        self.config.ocr_model,
                        attn_implementation="eager",
                        **model_kwargs,
                    )
            else:
                # Use eager attention (standard)
                self._model = AutoModel.from_pretrained(
                    self.config.ocr_model,
                    attn_implementation="eager",
                    **model_kwargs,
                )

            # Move to device if not already there (for CPU fallback)
            if device_map is None and self._device != "cpu":
                self._model = self._model.to(self._device)

            self._model.eval()
            
            logger.info(f"DeepSeek-OCR initialized successfully")
            logger.info(f"PyTorch version: {torch.__version__}")
            try:
                import transformers
                logger.info(f"Transformers version: {transformers.__version__}")
            except:
                pass

        except ImportError as e:
            error_msg = str(e)
            if "flash_attn" in error_msg.lower():
                raise OCRError(
                    "Flash Attention library not installed. Install with: pip install flash-attn --no-build-isolation",
                    details={"error": error_msg},
                )
            raise OCRError(
                "Failed to import required libraries. Install with: pip install deepcompress[gpu]",
                details={"error": error_msg},
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to initialize OCR model on {self._device}: {error_msg}")

            # Try fallback to CPU if CUDA failed
            if self._device.startswith("cuda") and "CUDA" in error_msg:
                logger.info("CUDA initialization failed, trying CPU fallback...")
                try:
                    self._device = "cpu"
                    self.config.ocr_device = "cpu"

                    # Retry initialization with CPU
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.config.ocr_model,
                        revision=self.config.ocr_model_revision,
                        trust_remote_code=True,
                    )

                    if self._tokenizer.pad_token is None:
                        if self._tokenizer.eos_token is not None:
                            self._tokenizer.pad_token = self._tokenizer.eos_token
                        else:
                            self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                    # Use CPU model
                    model_kwargs = {
                        "torch_dtype": torch.float32,  # Use float32 for CPU
                        "trust_remote_code": True,
                        "revision": self.config.ocr_model_revision,
                        "low_cpu_mem_usage": True,
                    }

                    self._model = AutoModel.from_pretrained(
                        self.config.ocr_model,
                        **model_kwargs,
                    )

                    self._model.eval()
                    logger.info("Successfully initialized DeepSeek-OCR on CPU")

                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
                    raise GPUError(
                        "Failed to initialize OCR model on both GPU and CPU",
                        details={"gpu_error": error_msg, "cpu_error": str(cpu_e)},
                    )
            else:
                raise GPUError(
                    "Failed to initialize OCR model",
                    details={"device": self._device, "error": error_msg},
                )

    async def extract(
        self,
        file_path: str,
        document_id: Optional[str] = None,
    ) -> ExtractedDocument:
        """
        Extract document content using DeepSeek-OCR.

        Args:
            file_path: Path to document (PDF or image)
            document_id: Optional document ID (generated if None)

        Returns:
            ExtractedDocument with extracted entities, tables, and text

        Raises:
            OCRError: If extraction fails
            GPUError: If GPU operations fail
        """
        if self._model is None:
            await self.initialize()

        start_time = time.time()

        try:
            images = await self._load_images(file_path)

            if document_id is None:
                document_id = self._generate_document_id(file_path)

            # Process pages in batches for better performance
            pages = await self._extract_pages_in_batches(images)

            processing_time_ms = (time.time() - start_time) * 1000

            return ExtractedDocument(
                document_id=document_id,
                page_count=len(pages),
                mode=self.config.ocr_mode,
                pages=pages,
                metadata={
                    "processing_time_ms": processing_time_ms,
                    "model": self.config.ocr_model,
                    "device": self._device,
                },
            )

        except Exception as e:
            raise OCRError(
                f"Failed to extract document: {file_path}",
                details={"error": str(e)},
            )

    async def _extract_pages_in_batches(self, images: list[Any]) -> list[Page]:
        """
        Extract pages in configurable batches for better performance with progress tracking.
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of extracted Page objects in original order
        """
        import logging
        
        logger = logging.getLogger(__name__)
        batch_size = self.config.ocr_batch_size
        total_pages = len(images)
        
        logger.info(f"Processing {total_pages} pages sequentially (to avoid output mixing)")
        
        all_pages = []
        
        # Add progress logging
        logger.info(f"Starting OCR extraction: {total_pages} pages")
        
        try:
            # Import tqdm for progress tracking (optional)
            from tqdm.auto import tqdm
            use_progress_bar = True
        except ImportError:
            use_progress_bar = False
            logger.debug("tqdm not available, progress bar disabled")
        
        # Setup progress bar with explicit file output to stdout
        pbar = None
        if use_progress_bar:
            import sys
            pbar = tqdm(
                total=total_pages,
                desc="OCR Processing",
                unit=" page",
                position=0,
                leave=True,
                file=sys.stdout,
                ncols=80
            )
        
        # Track skipped pages
        skipped_pages = []
        
        try:
            # Process pages sequentially to avoid output mixing
            # Each page processes independently with isolated stdout/stderr
            for page_idx, (image, page_num) in enumerate(zip(images, range(1, total_pages + 1)), 1):
                logger.info(f"Processing page {page_num}/{total_pages}...")
                
                try:
                    # Process page (async but sequential execution)
                    page_result = await self._extract_page(image, page_num)
                    all_pages.append(page_result)
                    
                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                        pbar.refresh()
                    
                    logger.info(f"✓ Page {page_num}/{total_pages} completed")
                    
                except Exception as e:
                    # Check if we should skip failed pages or raise error
                    if self.config.ocr_skip_failed_pages:
                        # Skip this page and continue processing
                        logger.warning(f"⚠ Skipping page {page_num} after multiple failed attempts: {e}")
                        skipped_pages.append(page_num)
                        
                        # Update progress bar to show we're moving on
                        if pbar is not None:
                            pbar.update(1)
                            pbar.refresh()
                        
                        # Continue processing remaining pages
                        continue
                    else:
                        # Original behavior: raise error and stop processing
                        logger.error(f"Failed to process page {page_num}: {e}")
                        if pbar is not None:
                            pbar.close()
                        raise OCRError(
                            f"Failed to extract page {page_num}",
                            details={"error": str(e)},
                        )
        
        finally:
            if pbar is not None:
                pbar.close()
        
        # Log summary of processing
        if skipped_pages:
            logger.warning(f"⚠ Processing completed with {len(skipped_pages)} page(s) skipped: {skipped_pages}")
            logger.info(f"✓ Successfully processed {len(all_pages)}/{total_pages} pages")
        else:
            logger.info(f"✓ All {total_pages} pages processed successfully")
        
        # If all pages failed, raise an error
        if not all_pages:
            raise OCRError(
                f"Failed to extract any pages from document (all {total_pages} pages failed)",
                details={"skipped_pages": skipped_pages},
            )
        
        return all_pages
    
    async def _load_images(self, file_path: str) -> list[Any]:
        """
        Load images from PDF or image file.

        Args:
            file_path: Path to file

        Returns:
            List of PIL Images
        """
        from PIL import Image

        path = Path(file_path)

        if path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path

                loop = asyncio.get_event_loop()
                images = await loop.run_in_executor(
                    None,
                    lambda: convert_from_path(
                        str(path),
                        dpi=300,
                        fmt="png",
                    ),
                )
                return images
            except ImportError:
                raise OCRError(
                    "pdf2image not installed. Install with: pip install deepcompress[gpu]"
                )
        else:
            image = Image.open(path).convert("RGB")
            return [image]

    async def _extract_page(self, image: Any, page_number: int) -> Page:
        """
        Extract single page using DeepSeek-OCR with optimized inference.

        Args:
            image: PIL Image
            page_number: Page number (1-indexed)

        Returns:
            Page with extracted raw text
        """
        import torch
        import tempfile
        import os
        import logging
        import warnings
        
        logger = logging.getLogger(__name__)
        logger.debug(f"Extracting page {page_number} (image size: {image.size})")

        tmp_dir = tempfile.mkdtemp()
        tmp_image_path = os.path.join(tmp_dir, 'page.png')
        image.save(tmp_image_path, format='PNG')
        
        result_text = None
        retry_count = 0
        max_retries = self.config.ocr_retry_attempts
        
        try:
            mode_config = {
                "small": {"base_size": 640, "image_size": 640},
                "base": {"base_size": 1024, "image_size": 640},
                "large": {"base_size": 1280, "image_size": 640},
            }
            config = mode_config.get(self.config.ocr_mode, mode_config["small"])
            
            # Optimized prompt that reduces hallucinations
            prompt = "<image>\nExtract all text exactly as shown:"
            
            output_dir = os.path.join(tmp_dir, f'output_page_{page_number}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Retry logic for robust inference
            while retry_count <= max_retries and result_text is None:
                try:
                    logger.debug(f"Page {page_number}: Starting inference")
                    logger.info(f"Processing page {page_number}...")
                    
                    # Suppress model's debug prints BEFORE calling executor
                    # Do this at Python level to catch all prints
                    import sys
                    import os
                    from io import StringIO
                    
                    # Create a wrapper function that captures stdout AND suppresses it
                    captured_output = []
                    
                    def run_inference_with_suppression():
                        import sys
                        import os
                        from io import StringIO
                        
                        # Capture stdout/stderr
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        stdout_capture = StringIO()
                        stderr_capture = StringIO()
                        
                        # Redirect stdout/stderr at file descriptor level too
                        original_stdout_fd = os.dup(1)
                        original_stderr_fd = os.dup(2)
                        devnull_fd = os.open(os.devnull, os.O_WRONLY)
                        
                        try:
                            # Set Python-level stdout/stderr
                            sys.stdout = stdout_capture
                            sys.stderr = stderr_capture
                            
                            # Also redirect file descriptors (for C-level prints)
                            os.dup2(devnull_fd, 1)
                            os.dup2(devnull_fd, 2)
                            
                            # Call model inference
                            result = self._model.infer(
                                self._tokenizer,
                                prompt=prompt,
                                image_file=tmp_image_path,
                                output_path=output_dir,
                                base_size=config["base_size"],
                                image_size=config["image_size"],
                                crop_mode=True,
                                save_results=False,
                                test_compress=False,
                            )
                            
                            # Get captured output
                            stdout_text = stdout_capture.getvalue()
                            stderr_text = stderr_capture.getvalue()
                            
                            # Store captured output
                            captured_output.append({
                                'result': result,
                                'stdout': stdout_text,
                                'stderr': stderr_text
                            })
                            
                            return result
                        finally:
                            # Restore Python-level stdout/stderr
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr
                            
                            # Restore file descriptors
                            os.dup2(original_stdout_fd, 1)
                            os.dup2(original_stderr_fd, 2)
                            os.close(original_stdout_fd)
                            os.close(original_stderr_fd)
                            os.close(devnull_fd)
                    
                    # Call in executor with suppression wrapper
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, run_inference_with_suppression)
                    
                    # Check captured output for text (model might print instead of return)
                    captured = captured_output[0] if captured_output else {}
                    captured_text = captured.get('stdout', '') or captured.get('stderr', '')
                    
                    # Use captured text if result is empty but we have captured output
                    if (not result or (isinstance(result, str) and not result.strip())) and captured_text.strip():
                        # Extract text from captured output (skip debug lines)
                        lines = captured_text.split('\n')
                        text_lines = []
                        skip_patterns = ['BASE:', 'PATCHES:', 'torch.Size', '=====']
                        for line in lines:
                            if not any(pattern in line for pattern in skip_patterns):
                                if line.strip():
                                    text_lines.append(line.strip())
                        if text_lines:
                            result = '\n'.join(text_lines)
                            logger.debug(f"Page {page_number}: Extracted text from captured stdout ({len(result)} chars)")
                    
                    # Process result - don't print it, just store it
                    if result is not None and isinstance(result, str) and result.strip():
                        result_text = result.strip()
                        logger.debug(f"Page {page_number}: Model returned {len(result_text)} chars directly")
                        logger.info(f"✓ Page {page_number} completed ({len(result_text)} chars)")
                        break  # Success - exit retry loop
                    
                    # Fallback: check output files only if direct return failed
                    if not result_text:
                        logger.debug(f"Page {page_number}: Checking output files (attempt {retry_count + 1})")
                        result_text = self._read_output_files(output_dir, page_number, logger)
                        
                        if result_text:
                            logger.info(f"Page {page_number}: Extracted {len(result_text)} chars from output file")
                            logger.info(f"✓ Page {page_number} completed ({len(result_text)} chars)")
                            break  # Success - exit retry loop
                    
                    # If we get here, result is empty - retry
                    if not result_text:
                        logger.warning(f"Page {page_number}: Empty result on attempt {retry_count + 1}, retrying...")
                        retry_count += 1
                        if retry_count <= max_retries:
                            await asyncio.sleep(0.5 * retry_count)
                        else:
                            break  # Exit loop, will raise error below
                    
                except Exception as infer_error:
                    logger.warning(f"Page {page_number} inference attempt {retry_count + 1} failed: {infer_error}")
                    retry_count += 1
                    
                    if retry_count <= max_retries:
                        await asyncio.sleep(0.5 * retry_count)
                    else:
                        raise
            
            # Validate result and handle empty pages gracefully
            if not result_text or result_text.strip() in ["", "None", "null"]:
                logger.warning(
                    f"Page {page_number}: OCR returned empty result after {retry_count + 1} attempts. "
                    f"This might be a blank page or contain only non-textual content. Returning empty text for this page."
                )
                result_text = ""  # Normalize to empty string
            
            # Apply hallucination filters (safe for empty strings)
            result_text = self._filter_prompt_hallucinations(result_text, page_number, logger)
            result_text = self._detect_and_fix_repetitions(result_text, page_number, logger)
            
            # Final validation: check for very short, potentially junk text
            if result_text and len(result_text) < self.config.min_ocr_text_length:
                logger.warning(
                    f"Page {page_number}: OCR produced insufficient text ({len(result_text)} chars). "
                    f"The content may be unreliable. Min length is {self.config.min_ocr_text_length}."
                )
            
        except Exception as e:
            logger.error(f"Error during model inference on page {page_number}: {str(e)}", exc_info=True)
            raise
        finally:
            import shutil
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            
            if self._device.startswith("cuda"):
                try:
                    import torch
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                except Exception as cache_e:
                    logger.debug(f"Could not clear GPU cache: {cache_e}")

        estimated_tokens = len(result_text.split()) * 1.3

        return Page(
            page_number=page_number,
            layout="multi_column",
            entities=[],
            tables=[],
            raw_text=result_text,
            metadata={"vision_tokens": int(estimated_tokens)},
        )


    def _read_output_files(self, output_dir: str, page_number: int, logger: Any) -> str or None:
        """
        Read OCR output from files as a fallback mechanism.
        
        Args:
            output_dir: Directory containing output files
            page_number: Page number for logging
            logger: Logger instance
            
        Returns:
            Extracted text or None if no valid output found
        """
        import os
        
        if not os.path.exists(output_dir):
            return None
        
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Skip image files
                if os.path.isfile(file_path) and not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    output_files.append(file_path)
        
        # Sort by modification time (most recent first)
        output_files.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
        
        for output_file in output_files:
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                    if content and len(content) > 10 and content not in ["None", "null", ""]:
                        logger.debug(f"Found OCR output in file: {output_file} ({len(content)} chars)")
                        return content
            except Exception as read_e:
                logger.debug(f"Could not read {output_file}: {read_e}")
                continue
        
        return None

    def _filter_prompt_hallucinations(self, text: str, page_number: int, logger: Any) -> str:
        """
        Filter out prompt-based hallucinations where the model repeats instructions.
        
        DeepSeek-OCR has a known issue where it repeats the prompt text instead of
        extracting actual content. This method aggressively filters such hallucinations.
        
        Args:
            text: The OCR output text
            page_number: Page number for logging
            logger: Logger instance
            
        Returns:
            Cleaned text with prompt hallucinations removed
        """
        if not text:
            return text
        
        # Common hallucination phrases that indicate the model is repeating instructions
        hallucination_phrases = [
            "Output only the extracted text",
            "without repetition or elaboration",
            "Input only the extracted text",
            "Extract the text content",
            "Focus on accuracy and completeness",
            "without repetition",
            "or elaboration",
            "and elaboration",
        ]
        
        # Check if the text contains excessive repetition of hallucination phrases
        for phrase in hallucination_phrases:
            phrase_count = text.lower().count(phrase.lower())
            if phrase_count > 3:  # More than 3 occurrences = likely hallucination
                logger.warning(f"Page {page_number}: Detected prompt hallucination ('{phrase}' repeated {phrase_count} times)")
                
                # Find where the hallucination starts (first occurrence)
                phrase_lower = phrase.lower()
                text_lower = text.lower()
                first_idx = text_lower.find(phrase_lower)
                
                if first_idx > 50:  # If there's real content before the hallucination
                    # Truncate at the hallucination point
                    text = text[:first_idx].strip()
                    logger.warning(f"Page {page_number}: Truncated at hallucination start (position {first_idx})")
                else:
                    # If hallucination is at the start, try to find any real content
                    lines = text.split('\n')
                    clean_lines = []
                    for line in lines:
                        line_lower = line.lower()
                        # Skip lines that are mostly hallucination phrases
                        has_hallucination = any(hp.lower() in line_lower for hp in hallucination_phrases)
                        if not has_hallucination and len(line.strip()) > 10:
                            clean_lines.append(line)
                        if len(clean_lines) >= 50:  # Max 50 clean lines
                            break
                    
                    if clean_lines:
                        text = '\n'.join(clean_lines)
                        logger.warning(f"Page {page_number}: Extracted {len(clean_lines)} clean lines from hallucinated output")
                    else:
                        text = ""
                        logger.error(f"Page {page_number}: Entire output appears to be hallucination, returning empty")
                
                break  # Stop after first detected hallucination pattern
        
        # Hard length limit to prevent infinite generation
        max_length = 50000  # 50K characters should be enough for any page
        if len(text) > max_length:
            logger.warning(f"Page {page_number}: Text exceeds {max_length} chars, truncating")
            text = text[:max_length]
        
        return text
    
    def _detect_and_fix_repetitions(self, text: str, page_number: int, logger: Any) -> str:
        """
        Detect and fix repetitive hallucinations in OCR output.
        
        Args:
            text: The OCR output text
            page_number: Page number for logging
            logger: Logger instance
            
        Returns:
            Cleaned text with repetitions removed
        """
        import re
        
        # If text is excessively long (> 50K chars), likely a hallucination
        max_reasonable_length = 50000
        if len(text) > max_reasonable_length:
            logger.warning(f"Page {page_number}: Text too long ({len(text)} chars), truncating to {max_reasonable_length}")
            text = text[:max_reasonable_length]
        
        # Detect patterns like "[123]" repeated many times
        # This catches the "College Algebra [1]...[2]...[3]..." pattern
        bracket_pattern = r'\[(\d+)\]'
        bracket_matches = re.findall(bracket_pattern, text)
        
        if len(bracket_matches) > 50:  # If more than 50 numbered references, likely hallucination
            logger.warning(f"Page {page_number}: Detected {len(bracket_matches)} numbered references, likely hallucination")
            # Find where the repetition starts (usually after legitimate content)
            # Look for the first occurrence of a pattern that repeats 3+ times
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if i > 0 and i < len(sentences) - 2:
                    # If this sentence is very similar to the next one, truncate here
                    if len(sentence) > 20:  # Only check substantial sentences
                        next_similar_count = sum(1 for s in sentences[i+1:i+5] if self._similarity(sentence, s) > 0.7)
                        if next_similar_count >= 2:
                            logger.warning(f"Page {page_number}: Found repetition starting at sentence {i}, truncating")
                            text = '.'.join(sentences[:i+1])
                            break
        
        # Detect repeated phrases (same phrase appearing 5+ times in a row)
        words = text.split()
        if len(words) > 100:
            # Check for phrases that repeat
            for phrase_len in range(5, 20):  # Check phrases of 5-20 words
                for i in range(len(words) - phrase_len * 3):
                    phrase = ' '.join(words[i:i+phrase_len])
                    next_phrase = ' '.join(words[i+phrase_len:i+phrase_len*2])
                    if self._similarity(phrase, next_phrase) > 0.8:
                        # Count how many times it repeats
                        repeat_count = 1
                        pos = i + phrase_len
                        while pos < len(words) - phrase_len:
                            test_phrase = ' '.join(words[pos:pos+phrase_len])
                            if self._similarity(phrase, test_phrase) > 0.8:
                                repeat_count += 1
                                pos += phrase_len
                            else:
                                break
                        
                        if repeat_count >= 3:
                            logger.warning(f"Page {page_number}: Found phrase repeating {repeat_count} times, truncating")
                            text = ' '.join(words[:i])
                            break
        
        return text
    
    def _similarity(self, s1: str, s2: str) -> float:
        """
        Calculate simple similarity between two strings (0-1).
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Similarity score (0-1)
        """
        if not s1 or not s2:
            return 0.0
        
        # Simple character-based similarity
        s1_clean = s1.lower().strip()
        s2_clean = s2.lower().strip()
        
        if s1_clean == s2_clean:
            return 1.0
        
        # Count matching characters in same positions
        max_len = max(len(s1_clean), len(s2_clean))
        if max_len == 0:
            return 0.0
        
        matches = sum(1 for i in range(min(len(s1_clean), len(s2_clean))) 
                     if s1_clean[i] == s2_clean[i])
        
        return matches / max_len
    

    def _generate_document_id(self, file_path: str) -> str:
        """
        Generate unique document ID from file path.

        Args:
            file_path: Path to file

        Returns:
            Document ID (hash of file path)
        """
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    def _apply_transformers_compatibility_patch(self) -> None:
        """
        Apply compatibility patches for newer transformers versions.
        
        This fixes multiple compatibility issues:
        1. LlamaFlashAttention2 import error in older transformers
        2. DynamicCache.get_max_length() -> get_seq_length() API change
        3. LlamaAttention position_embeddings argument requirement in v4.46+
        """
        try:
            from transformers.models.llama import modeling_llama
            import inspect
            
            # Patch 1: Fix LlamaFlashAttention2 missing in newer transformers
            if not hasattr(modeling_llama, 'LlamaFlashAttention2'):
                if hasattr(modeling_llama, 'LlamaAttention'):
                    modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
                elif hasattr(modeling_llama, 'LlamaSdpaAttention'):
                    modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaSdpaAttention
                else:
                    class LlamaFlashAttention2Fallback:
                        """Fallback class for missing LlamaFlashAttention2"""
                        pass
                    modeling_llama.LlamaFlashAttention2 = LlamaFlashAttention2Fallback
            
            # Patch 3: Fix position_embeddings requirement in transformers >= 4.46
            # This wraps LlamaAttention classes to make position_embeddings optional
            for attention_class_name in ['LlamaAttention', 'LlamaFlashAttention2', 'LlamaSdpaAttention']:
                if hasattr(modeling_llama, attention_class_name):
                    original_class = getattr(modeling_llama, attention_class_name)
                    if not hasattr(original_class, '_deepcompress_patched'):
                        original_forward = original_class.forward
                        
                        # Check if position_embeddings is already in the signature
                        sig = inspect.signature(original_forward)
                        accepts_position_embeddings = 'position_embeddings' in sig.parameters

                        def create_wrapped_forward(orig_forward):
                            def wrapped_forward(self, hidden_states, attention_mask=None,
                                                position_ids=None, past_key_value=None,
                                                output_attentions=False, use_cache=False,
                                                cache_position=None, position_embeddings=None, **kwargs):
                                rotary_emb = getattr(self, 'rotary_emb', None)
                                computed_position_embeddings = position_embeddings
                                
                                if rotary_emb is not None and accepts_position_embeddings and computed_position_embeddings is None:
                                    import torch

                                    seq_len = hidden_states.size(1)
                                    pos_ids = position_ids
                                    if pos_ids is None:
                                        if cache_position is not None:
                                            pos_ids = cache_position.view(1, -1).to(hidden_states.device)
                                        else:
                                            past_length = 0
                                            if past_key_value is not None:
                                                try:
                                                    if hasattr(past_key_value, "get_usable_length"):
                                                        past_length = int(past_key_value.get_usable_length(seq_len) or 0)
                                                    elif hasattr(past_key_value, "seen_tokens"):
                                                        past_length = int(past_key_value.seen_tokens or 0)
                                                except Exception:
                                                    past_length = 0
                                            pos_ids = torch.arange(
                                                past_length,
                                                past_length + seq_len,
                                                dtype=torch.long,
                                                device=hidden_states.device,
                                            ).unsqueeze(0)

                                    try:
                                        heads = getattr(self, "num_key_value_heads", getattr(self, "num_heads", 0))
                                        head_dim = getattr(self, "head_dim", 0)
                                        if heads and head_dim:
                                            dummy = torch.zeros(
                                                hidden_states.size(0),
                                                heads,
                                                seq_len,
                                                head_dim,
                                                dtype=hidden_states.dtype,
                                                device=hidden_states.device,
                                            )
                                            import inspect as _inspect
                                            rotary_callable = getattr(rotary_emb, "forward", rotary_emb)
                                            rotary_sig = _inspect.signature(rotary_callable)
                                            if "cache_position" in rotary_sig.parameters:
                                                cos, sin = rotary_emb(dummy, pos_ids, cache_position=cache_position)
                                            else:
                                                cos, sin = rotary_emb(dummy, pos_ids)
                                            computed_position_embeddings = (cos, sin)
                                    except Exception as e:
                                        # Log the error but continue - the model may work without it
                                        import warnings
                                        warnings.warn(f"Failed to compute position embeddings: {e}")

                                kwargs_payload = {
                                    "hidden_states": hidden_states,
                                    "attention_mask": attention_mask,
                                    "position_ids": position_ids,
                                    "past_key_value": past_key_value,
                                    "output_attentions": output_attentions,
                                    "use_cache": use_cache,
                                    "cache_position": cache_position,
                                }
                                kwargs_payload.update(kwargs)
                                # Only add position_embeddings if we successfully computed it
                                if accepts_position_embeddings and computed_position_embeddings is not None:
                                    kwargs_payload["position_embeddings"] = computed_position_embeddings

                                return orig_forward(self, **kwargs_payload)

                            return wrapped_forward
                        
                        original_class.forward = create_wrapped_forward(original_forward)
                        original_class._deepcompress_patched = True
        except (ImportError, AttributeError) as e:
            # Log but don't fail - patches are optional
            import warnings
            warnings.warn(f"Could not apply all compatibility patches: {e}")
        
        # Patch 2: Fix DynamicCache API change (get_max_length -> get_seq_length)
        try:
            from transformers.cache_utils import DynamicCache
            
            # Check if get_max_length is missing but get_seq_length exists
            if not hasattr(DynamicCache, 'get_max_length') and hasattr(DynamicCache, 'get_seq_length'):
                # Add get_max_length as an alias to get_seq_length
                DynamicCache.get_max_length = DynamicCache.get_seq_length

            # Add seen_tokens compatibility shim for newer transformers
            if not hasattr(DynamicCache, 'seen_tokens'):
                def _get_seen_tokens(self):
                    """
                    Provide a unified accessor for the number of cached tokens.
                    """
                    try:
                        if hasattr(self, 'get_seq_length'):
                            return self.get_seq_length()
                        if hasattr(self, 'get_max_length'):
                            return self.get_max_length()
                    except Exception:
                        # Fall back to inspecting the key cache shape
                        pass

                    cache = getattr(self, "key_value_cache", None) or getattr(self, "key_cache", None)
                    if cache:
                        first = None
                        if isinstance(cache, dict):
                            first = next(iter(cache.values()), None)
                        elif isinstance(cache, (list, tuple)) and cache:
                            first = cache[0]
                        if isinstance(first, (list, tuple)) and first:
                            first = first[0]
                        if hasattr(first, "shape"):
                            return first.shape[-2]
                    return getattr(self, "_deepcompress_seen_tokens", 0)

                def _set_seen_tokens(self, value):
                    """
                    Map seen_tokens assignments to the new API when available.
                    """
                    if hasattr(self, 'set_seq_length'):
                        try:
                            self.set_seq_length(value)
                            return
                        except Exception:
                            pass
                    if hasattr(self, '_set_seen_tokens'):
                        try:
                            self._set_seen_tokens(value)
                            return
                        except Exception:
                            pass
                    self._deepcompress_seen_tokens = value

                DynamicCache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)

            # Add get_usable_length shim (renamed in newer transformers)
            if not hasattr(DynamicCache, 'get_usable_length'):
                def _get_usable_length(self, seq_length=None):
                    """
                    Return usable token length expected by older DeepSeek builds.
                    """
                    try:
                        if hasattr(self, 'get_seq_length'):
                            return self.get_seq_length()
                        if hasattr(self, 'get_max_length'):
                            return self.get_max_length()
                    except Exception:
                        pass

                    length = None
                    if hasattr(self, 'seen_tokens'):
                        try:
                            length = self.seen_tokens
                        except Exception:
                            length = None

                    if length is None:
                        cache = getattr(self, "key_value_cache", None) or getattr(self, "key_cache", None)
                        if cache:
                            first = None
                            if isinstance(cache, dict):
                                first = next(iter(cache.values()), None)
                            elif isinstance(cache, (list, tuple)) and cache:
                                first = cache[0]
                            if isinstance(first, (list, tuple)) and first:
                                first = first[0]
                            if hasattr(first, "shape"):
                                length = first.shape[-2]

                    if length is None:
                        length = getattr(self, "_deepcompress_seen_tokens", 0)

                    if seq_length is not None:
                        return min(length, seq_length)
                    return length

                DynamicCache.get_usable_length = _get_usable_length
        except (ImportError, AttributeError):
            pass

    async def extract_batch(
        self,
        file_paths: list[str],
    ) -> list[ExtractedDocument]:
        """
        Extract multiple documents in batch.

        Args:
            file_paths: List of file paths

        Returns:
            List of ExtractedDocuments
        """
        tasks = [self.extract(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        documents = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise OCRError(
                    f"Batch extraction failed for {file_paths[i]}",
                    details={"error": str(result)},
                )
            documents.append(result)

        return documents

