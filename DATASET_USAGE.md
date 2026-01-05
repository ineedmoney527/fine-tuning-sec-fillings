# Example usage instructions for generate_dataset.py

## Prerequisites
1. Set DeepSeek API key:
   ```bash
   export DEEPSEEK_API_KEY="your-deepseek-api-key"
   ```
   Get your API key from: https://platform.deepseek.com/

2. Ensure you have processed Markdown files in `data/processed/`

## Run the dataset generator
```bash
python -m src.generate_dataset
```

## Output
- Creates `data/train.jsonl` in ChatML format
- Each line is a JSON object with conversations array
- Format compatible with Qwen3 fine-tuning

## Example output format
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a financial analyst..."
    },
    {
      "role": "user",
      "content": "[Markdown content from 10-K]"
    },
    {
      "role": "assistant",
      "content": "{\"Revenue\": 391035000000, \"Operating Income\": 123216000000, ...}"
    }
  ]
}
```
