# Financial Extraction Engine

Fine-tune Qwen3-8B-Instruct to extract structured JSON metrics from SEC 10-K filings.

## Tech Stack

- **Data pipeline**: Python, `sec-api`, `docling` (HTML→Markdown)
- **Training**: `unsloth` (fast QLoRA), Hugging Face `trl`
- **Inference**: `outlines` or `pydantic` for structured generation
- **Prompt format**: ChatML (`<|im_start|>...`)
- **Code standards**: Modular, type-hinted, `loguru` for logging

## Project Structure

```
financial-extraction-engine/
├── data/
│   ├── raw/              # Raw 10-K HTML files (Item 8)
│   ├── processed/        # Clean Markdown files
│   └── train.jsonl       # Training dataset (ChatML format)
├── src/                  # Source code modules (coming soon)
├── logs/                 # Application logs
├── download_filings.py   # SEC 10-K downloader
├── process_data.py       # HTML to Markdown converter
├── generate_dataset.py   # Dataset generator (DeepSeek-V3 teacher)
├── requirements.txt      # Python dependencies
├── DATASET_USAGE.md      # Dataset generation guide
└── README.md             # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `docling` is the primary HTML→Markdown converter. If installation fails (heavy dependencies), the script will automatically fall back to `markdownify`.

### 2. Set API Keys

**SEC API** (for downloading filings):

```bash
export SEC_API_KEY="your-sec-api-key"
```

Get your key from [sec-api.io](https://sec-api.io/)

**DeepSeek API** (for dataset generation):

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

Get your key from [platform.deepseek.com](https://platform.deepseek.com/)

## Usage

### 1. Download 10-K Item 8 Filings

Download Item 8 (Financial Statements) from 10-K filings for AAPL, MSFT, TSLA, NVDA, GOOGL (2023-2025):

```bash
python -m src.download
```

**Output**: HTML files saved to `data/raw/{ticker}_{year}.html`

**Features**:
- ✅ Automatic rate limiting (0.3s between API calls)
- ✅ Comprehensive error handling and retry logic
- ✅ Detailed logging to both console and `logs/` directory
- ✅ Progress tracking and download statistics

### 2. Convert HTML to Clean Markdown

Process raw HTML files into clean Markdown with table preservation:

```bash
python -m src.process_data
```

**Output**: Markdown files saved to `data/processed/{ticker}_{year}.md`

**Features**:
- ✅ **Table preservation**: Converts HTML tables to Markdown pipe syntax
- ✅ **Noise removal**: Removes lines with <5 words (configurable)
- ✅ **Dual converter support**: Uses `docling` (primary) or `markdownify` (fallback)
- ✅ **Smart cleaning**: Preserves headers, tables, and meaningful content

### 3. Generate Fine-Tuning Dataset

Use DeepSeek-V3 as a teacher model to label the data:

```bash
# Set DeepSeek API key
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Generate dataset
python -m src.generate_dataset
```

**Output**: ChatML-formatted JSONL file saved to `data/train.jsonl`

**Features**:
- ✅ **Teacher-student paradigm**: Uses DeepSeek-V3 to label data for Qwen3
- ✅ **Number normalization**: Converts millions/billions to full values
- ✅ **ChatML format**: Compatible with Qwen3 fine-tuning
- ✅ **Structured extraction**: Extracts 6 key financial metrics
- ✅ **Rate limiting**: Respects API limits with delays

**Extracted metrics**:
- Revenue
- Operating Income  
- Net Income
- Total Assets
- Cash and Cash Equivalents
- Earnings Per Share (Diluted)

## Configuration

### src.download and src.process_data
- **Tickers/Years**: Modify variables in `src/download.py`
- **Rate limit**: Adjust `RATE_LIMIT_DELAY` in `src/download.py`
- **Min words**: Adjust `MIN_WORDS_PER_LINE` in `src/process_data.py`

### src.generate_dataset
- **Teacher model**: Change `MODEL_NAME` (default: `deepseek-chat`)
- **Target metrics**: Modify `TARGET_METRICS` list
- **System prompt**: Customize `SYSTEM_PROMPT` for different extraction rules

## Fine-Tuning with QLoRA

### Prerequisites from requirements.txt
Install training dependencies (requires GPU with ~24GB VRAM):

```bash
pip install -r requirements.txt
```

### Training

Run the QLoRA fine-tuning script:

```bash
# Full training run (requires GPU)
python -m src.train --output_dir outputs/qwen3-8b-financial-lora

# Dry run with 2 steps (for testing)
python -m src.train --max_steps 2 --output_dir outputs/test_run
```

**Training Configuration:**
- Model: `unsloth/Qwen3-8B-Instruct-unsloth-bnb-4bit`
- LoRA rank: 16, alpha: 32
- Learning rate: 2e-4
- Epochs: 3

### Inference with Pydantic Validation

The inference module includes automatic retry logic for malformed JSON outputs:

```python
from src.inference import FinancialExtractor

# Load fine-tuned model
extractor = FinancialExtractor("outputs/qwen3-8b-financial-lora")

# Extract metrics (with automatic retry on validation errors)
result = extractor.extract(markdown_content)

# Access validated metrics
print(result.metrics.revenue)       # int or None
print(result.metrics.net_income)    # int or None
print(result.metrics.diluted_eps)   # float or None
```

**Features:**
- ✅ Pydantic validation for structured output
- ✅ Automatic retry with error feedback (max 3 attempts)
- ✅ Type coercion (strings → ints/floats)
- ✅ Null handling for missing data

## Project Status

1. ✅ **Data collection** with `sec-api`
2. ✅ **HTML→Markdown conversion** with `docling`
3. ✅ **Dataset generation** with DeepSeek-V3 teacher
4. ✅ **Fine-tuning pipeline** with `unsloth` and QLoRA
5. ✅ **Structured inference** with Pydantic validation + retry
6. 🔲 **Evaluation and deployment**

## License

MIT
